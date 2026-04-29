import argparse
import logging

from dash.development.base_component import ComponentRegistry
from dash import Dash, html

# TODO: this still imports all of kaolin, can we do better?
from kaolin.utils.log import default_log_setup, add_log_level_flag
from kaolin.visualize.dash.components import KaolinViewerInternal
from kaolin.visualize.dash import KaolinViewer

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sample application')
    parser.add_argument('--port', default=8080, type=int)
    add_log_level_flag(parser)
    args = parser.parse_args()

    default_log_setup(level=args.log_level)
    print(f'Component registry initial: {len(ComponentRegistry.registry)} components')
    # print(f'Component namespace_to_package: {[x for x in ComponentRegistry.namespace_to_package]}')

    app = Dash()

    my_viewer = KaolinViewer.convenience_init()

    app.layout = [
        html.H1(children='Sample App', style={'textAlign': 'center'}),
        *my_viewer.components
        #KaolinViewerInternal()
        # dcc.Slider(0, 20, 5,
        #            value=pt_size,
        #            id='pt-size'),
        # dcc.Graph(
        #     id='pt-graph',
        #     figure=fig
        # ),
    ]
    print(f'Static folder: {app.server.static_folder}')
    print(f'Static URL path: {app.server.static_url_path}')
    print(f'Component registry: {len(ComponentRegistry.registry)} ' + 
          f'{[x for x in ComponentRegistry.registry]}')
    print(f'Config: {app.config}')
    print(f'Registered paths: {app.registered_paths}')

    logger.info(f'-----> Now starting Dash App on port {args.port}')
    app.logger.setLevel(args.log_level)
    app.run(debug=False, port=args.port)  # if you want to debug, better debug=False, or else no stack trace