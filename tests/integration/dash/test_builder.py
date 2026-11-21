# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Running-app builder tests; browser case requires Playwright and Chrome."""

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import pytest
from tornado.testing import AsyncHTTPTestCase

from kaolin.visualize.dash.builder import SessionRegistry, WebappBuilder
from kaolin.visualize.dash.layout import StandardLayoutHelper


@dataclass
class Settings:
    size: int = 3
    opacity: float = 0.5


class TestWebappBuilder(AsyncHTTPTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        sessions = patch.object(SessionRegistry, '_entries', {})
        sessions.start()
        self.addCleanup(sessions.stop)
        super().setUp()

    def get_app(self):
        self.builder = WebappBuilder(debug=False, default_theme='', set_favicon=False)
        module = self.directory / 'example.js'
        module.write_text('export const answer = 42;')
        self.module_url = '/' + self.builder.add_static_file(str(module), 'modules/example.js')
        self.directory_url = '/' + self.builder.add_static_dir(str(self.directory), 'data')
        self.builder.add_importmap_item('example', self.module_url, 'builderExample')
        self.builder.add_raw_body_html('<div id="custom-html">Custom content</div>')
        helper = StandardLayoutHelper()
        controls = self.builder.add_user_settings(Settings(), 'brush')
        self.info = self.builder.user_setting_infos['brush']
        self.info.store.data = {'size': 4, 'opacity': 0.25}
        helper.add_main_content(controls)
        self.builder.set_layout_helper(helper)
        self.app, server = self.builder.build([])

        @self.app.server.route('/test-settings/<tab>')
        def settings(tab):
            session = SessionRegistry.get(tab)
            return asdict(session['brush']) if session else {}

        return server

    def test_static_routes_and_build(self):
        assert self.builder.default_user_session['brush'] == Settings(4, 0.25)
        response = self.fetch(self.module_url)
        assert response.code == 200 and response.body == b'export const answer = 42;'
        assert self.fetch(self.directory_url + '/example.js').body == response.body
        (self.directory / 'new.txt').write_text('new file')
        assert self.fetch(self.directory_url + '/new.txt').body == b'new file'
        assert self.fetch('/kaolin_custom/missing.txt').code == 404
        assert self.fetch('/_kaolin_assets/standard_layout.css').code == 200
        page = self.fetch('/').body.decode()
        assert '<div id="custom-html">Custom content</div>' in page
        assert '"example": "' + self.module_url + '"' in page
        with pytest.raises(AssertionError, match='only call build once'):
            self.builder.build([])

    def test_importmap_and_settings_across_tabs_and_reload(self):
        playwright = pytest.importorskip('playwright.sync_api')
        chrome = shutil.which('google-chrome') or shutil.which('chromium')
        url = self.get_url('/')
        fields = {spec.name: spec.control_id for spec in self.info.field_specs}

        def wait_settings(page, tab, expected):
            page.wait_for_function(
                """async ({tab, expected}) => {
                    const settings = await (await fetch('/test-settings/' + tab)).json();
                    return Object.entries(expected).every(([key, value]) => settings[key] === value);
                }""", arg={'tab': tab, 'expected': expected})

        def exercise_browser():
            with playwright.sync_playwright() as pw:
                browser = pw.chromium.launch(executable_path=chrome, headless=True)
                try:
                    context = browser.new_context()
                    first, second = context.new_page(), context.new_page()
                    for page in (first, second):
                        page.goto(url)
                        page.wait_for_function('window.builderExample?.answer === 42')
                        page.wait_for_function('window.kaolin?.util?.getTabUuid()')
                        playwright.expect(page.locator('#' + fields['size'])).to_have_value('4')
                    first_tab = first.evaluate('kaolin.util.getTabUuid()')
                    second_tab = second.evaluate('kaolin.util.getTabUuid()')
                    assert first_tab != second_tab
                    wait_settings(first, first_tab, {'size': 4, 'opacity': 0.25})
                    wait_settings(second, second_tab, {'size': 4, 'opacity': 0.25})
                    first.locator('#' + fields['size']).fill('7')
                    first.locator('#' + fields['size']).press('Tab')
                    wait_settings(first, first_tab, {'size': 7, 'opacity': 0.25})
                    first.locator('#' + fields['opacity']).fill('0.75')
                    first.locator('#' + fields['opacity']).press('Tab')
                    wait_settings(first, first_tab, {'size': 7, 'opacity': 0.75})
                    wait_settings(second, second_tab, {'size': 4, 'opacity': 0.25})
                    playwright.expect(second.locator('#' + fields['size'])).to_have_value('4')
                    first.reload()
                    playwright.expect(first.locator('#' + fields['size'])).to_have_value('7')
                    playwright.expect(first.locator('#' + fields['opacity'])).to_have_value('0.75')
                    assert first.evaluate('kaolin.util.getTabUuid()') == first_tab
                    wait_settings(first, first_tab, {'size': 7, 'opacity': 0.75})
                finally:
                    browser.close()

        # Keep Tornado serving while the synchronous browser client runs.
        self.io_loop.run_sync(lambda: asyncio.to_thread(exercise_browser), timeout=60)
