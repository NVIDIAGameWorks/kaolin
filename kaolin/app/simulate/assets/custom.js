// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Custom JS behavior for the Kaolin Simulate app.
// Renders a segment-list-style object list with collapsible transform controls.

(function () {
    const { defaultsFromSchema, BehaviorRegister, MessageHandlerBase } = kaolin.core.behavior;

    const _SimTags = {
        SELECT_OBJECTS:   'select_objects',
        SET_TRANSFORM:    'set_transform',
        REMOVE_OBJECT:    'remove_object',
        PATH_COMPLETE:    'path_complete',
        OBJECT_LIST:         'object_list',
        AVAILABLE_GAUSSIANS: 'available_gaussians',
        SIM_STATUS:          'sim_status',
        PATH_COMPLETIONS:    'path_completions',
        ERROR:               'error',
    };

    const SimControllerSchema = {
        listContainerId: {
            kind: 'string',
            default: 'object-list-container',
            uiBound: false,
            description: 'DOM id of the div that receives the dynamic object list.',
        },
        connectionId: {
            kind: 'string',
            default: 'main-ws',
            uiBound: false,
            description: 'WebSocket connection id used by this app.',
        },
    };

    // -----------------------------------------------------------------------
    // ObjectList: renders loaded objects as a segment-list-style Bootstrap
    // list-group. One row is "active" at a time; clicking it expands the
    // transform sliders. Mirrors the SegmentList pattern from the segment app.
    // -----------------------------------------------------------------------
    class ObjectList {
        constructor(elementId, onSend) {
            this.elementId = elementId;
            this.onSend    = onSend;   // (tag, content) → void
            this.items     = [];       // plain objects: {obj_id, name, translation, rotation_euler_deg}
            this.activeKey = null;     // obj_id of the expanded row, or null
            this._pendingPath = null;  // USD path awaiting prim selection
        }

        get element() { return document.getElementById(this.elementId); }

        setObjects(objects) {
            // Prune activeKey if that object was removed.
            const ids = new Set((objects || []).map(o => o.obj_id));
            if (this.activeKey !== null && !ids.has(this.activeKey)) {
                this.activeKey = null;
            }
            this.items = objects || [];
            this._render();
        }

        setActive(objId) {
            const next = (this.activeKey === objId) ? null : objId;
            this.activeKey = next;
            this._render();
        }

        // Build a Bootstrap icon button (matches the segment app helper).
        _iconButton({ icon, title, onClick, colorClass = 'text-body' }) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = `btn btn-link p-1 lh-1 fs-4 flex-shrink-0 text-decoration-none ${colorClass}`;
            btn.title = title;
            btn.setAttribute('aria-label', title);
            btn.innerHTML = `<i class="bi ${icon}"></i>`;
            btn.addEventListener('click', (e) => { e.stopPropagation(); onClick(); });
            return btn;
        }

        // Single labeled range slider with a live value readout.
        _makeSlider({ id, label, min, max, step, value, onChange }) {
            const wrap = document.createElement('div');
            wrap.className = 'mb-1';

            const header = document.createElement('div');
            header.className = 'd-flex justify-content-between';
            const lbl = document.createElement('small');
            lbl.className = 'text-muted';
            lbl.textContent = label;
            const val = document.createElement('small');
            val.id = id + '-val';
            val.textContent = (+value).toFixed(2);
            header.appendChild(lbl);
            header.appendChild(val);
            wrap.appendChild(header);

            const input = document.createElement('input');
            input.type  = 'range';
            input.className = 'form-range';
            input.id    = id;
            input.min   = String(min);
            input.max   = String(max);
            input.step  = String(step);
            input.value = String(value);
            input.addEventListener('input', () => {
                val.textContent = (+input.value).toFixed(2);
                onChange();
            });
            wrap.appendChild(input);
            return wrap;
        }

        // Read all 6 slider values for the given obj_id and send SET_TRANSFORM.
        _sendTransform(objId) {
            const t = [0, 1, 2].map(i => {
                const el = document.getElementById('obj-' + objId + '-t' + i);
                return el ? +el.value : 0;
            });
            const r = [0, 1, 2].map(i => {
                const el = document.getElementById('obj-' + objId + '-r' + i);
                return el ? +el.value : 0;
            });
            // Update local item state so a collapse+re-expand preserves current values
            // without waiting for the server to send back an updated object_list.
            const item = this.items.find(o => o.obj_id === objId);
            if (item) {
                item.translation = t;
                item.rotation_euler_deg = r;
                // Also update the collapsed subtitle immediately.
                const info = document.querySelector(
                    '#object-list-container .kaolin-seg-info[data-obj="' + objId + '"]');
                if (info) {
                    info.textContent = 'T(' + t.map(v => v.toFixed(2)).join(', ') + ')  '
                                     + 'R(' + r.map(v => v.toFixed(1)).join(', ') + '°)';
                }
            }
            this.onSend(_SimTags.SET_TRANSFORM, { obj_id: objId, translation: t, rotation_euler_deg: r });
        }

        _render() {
            const container = this.element;
            if (!container) return;

            if (this.items.length === 0) {
                container.innerHTML = '<p class="text-muted small px-1 mt-1">No objects loaded. Use "Load Object" to add one.</p>';
                return;
            }

            const list = document.createElement('ul');
            list.className = 'list-group kaolin-seg-container';

            this.items.forEach((item) => {
                const isActive = this.activeKey === item.obj_id;
                const entry = document.createElement('li');
                entry.className = 'list-group-item' + (isActive ? ' kaolin-seg-active' : '');
                entry.style.cursor = 'pointer';

                // ---- Collapsed row: name + position subtitle + delete icon ----
                const nameRow = document.createElement('div');
                nameRow.className = 'd-flex justify-content-between align-items-center gap-2';
                nameRow.addEventListener('click', () => this.setActive(item.obj_id));

                const labels = document.createElement('div');
                labels.className = 'flex-grow-1';
                labels.style.minWidth = '0';

                const name = document.createElement('div');
                name.className = 'fw-semibold text-truncate';
                name.textContent = item.name || ('Object ' + item.obj_id);
                name.title = item.name || '';
                labels.appendChild(name);

                const t = item.translation || [0, 0, 0];
                const r = item.rotation_euler_deg || [0, 0, 0];
                const info = document.createElement('div');
                info.className = 'kaolin-seg-info text-muted text-truncate';
                info.setAttribute('data-obj', String(item.obj_id));
                info.textContent = 'T(' + t.map(v => v.toFixed(2)).join(', ') + ')  '
                                 + 'R(' + r.map(v => v.toFixed(1)).join(', ') + '°)';
                labels.appendChild(info);

                nameRow.appendChild(labels);
                nameRow.appendChild(this._iconButton({
                    icon: 'bi-trash',
                    title: 'Remove object',
                    colorClass: 'text-danger',
                    onClick: () => this.onSend(_SimTags.REMOVE_OBJECT, { obj_id: item.obj_id }),
                }));
                entry.appendChild(nameRow);

                // ---- Expanded controls: 6 sliders (position + orientation) ----
                if (isActive) {
                    const controls = document.createElement('div');
                    controls.className = 'mt-2';

                    const tLabels = ['X', 'Y', 'Z'];
                    const onChange = () => this._sendTransform(item.obj_id);

                    const posHdr = document.createElement('div');
                    posHdr.className = 'small fw-semibold text-muted mb-1 mt-1';
                    posHdr.innerHTML = '<i class="bi bi-arrows-move me-1"></i>Position';
                    controls.appendChild(posHdr);
                    [0, 1, 2].forEach(i => {
                        controls.appendChild(this._makeSlider({
                            id: 'obj-' + item.obj_id + '-t' + i,
                            label: tLabels[i],
                            min: -5, max: 5, step: 0.05,
                            value: t[i],
                            onChange,
                        }));
                    });

                    const rotHdr = document.createElement('div');
                    rotHdr.className = 'small fw-semibold text-muted mb-1 mt-2';
                    rotHdr.innerHTML = '<i class="bi bi-arrow-repeat me-1"></i>Orientation (°)';
                    controls.appendChild(rotHdr);
                    [0, 1, 2].forEach(i => {
                        controls.appendChild(this._makeSlider({
                            id: 'obj-' + item.obj_id + '-r' + i,
                            label: tLabels[i],
                            min: -180, max: 180, step: 1,
                            value: r[i],
                            onChange,
                        }));
                    });

                    entry.appendChild(controls);
                }

                list.appendChild(entry);
            });

            container.replaceChildren(list);
        }

        // Show a prim-selection panel INSIDE the Load Object modal.
        // When the user confirms, the modal is closed and the sidebar list updates.
        showGaussianSelector(path, entries) {
            this._pendingPath = path;

            const selectorDiv = document.getElementById('modal-gaussian-selector');
            const pathSection = document.getElementById('modal-path-section');
            if (!selectorDiv) return;

            // Hide the path-input section and reveal the selector.
            if (pathSection) pathSection.style.display = 'none';
            selectorDiv.style.display = 'block';
            selectorDiv.innerHTML = '';

            const title = document.createElement('div');
            title.className = 'fw-semibold small mb-2';
            title.textContent = 'Multiple objects found — select which to load:';
            selectorDiv.appendChild(title);

            const checkList = document.createElement('div');
            checkList.className = 'mb-2';
            entries.forEach((entry) => {
                const row = document.createElement('div');
                row.className = 'd-flex align-items-center gap-2 mb-1';
                const cb = document.createElement('input');
                cb.type             = 'checkbox';
                cb.value            = entry.prim_path;
                cb.checked          = true;
                cb.id               = 'gs-sel-' + entry.prim_path.replace(/\//g, '_');
                cb.dataset.isSubset  = entry.is_subset ? 'true' : 'false';
                cb.dataset.parentPrim = entry.parent_prim || '';
                cb.dataset.name      = entry.name || entry.prim_path.split('/').pop();
                const lbl = document.createElement('label');
                lbl.htmlFor = cb.id;
                lbl.className = 'small';
                lbl.textContent = entry.name;
                row.appendChild(cb);
                row.appendChild(lbl);
                checkList.appendChild(row);
            });
            selectorDiv.appendChild(checkList);

            const btnRow = document.createElement('div');
            btnRow.className = 'd-flex gap-2 mt-1';

            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'btn btn-sm btn-outline-secondary flex-grow-1';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.addEventListener('click', () => this._resetModalToPathInput());

            const loadBtn = document.createElement('button');
            loadBtn.className = 'btn btn-sm btn-primary flex-grow-1';
            loadBtn.textContent = 'Load Selected';
            loadBtn.addEventListener('click', () => {
                const selected = Array.from(checkList.querySelectorAll('input:checked')).map(c => ({
                    prim_path:   c.value,
                    is_subset:   c.dataset.isSubset === 'true',
                    parent_prim: c.dataset.parentPrim || '',
                    name:        c.dataset.name || c.value.split('/').pop(),
                }));
                if (selected.length) {
                    this.onSend(_SimTags.SELECT_OBJECTS, { path: this._pendingPath, selected_entries: selected });
                }
                this._resetModalToPathInput();
                this._closeLoadModal();
            });

            btnRow.appendChild(cancelBtn);
            btnRow.appendChild(loadBtn);
            selectorDiv.appendChild(btnRow);
        }

        _resetModalToPathInput() {
            const selectorDiv = document.getElementById('modal-gaussian-selector');
            const pathSection = document.getElementById('modal-path-section');
            if (selectorDiv) { selectorDiv.style.display = 'none'; selectorDiv.innerHTML = ''; }
            if (pathSection) pathSection.style.display = 'block';
        }

        _closeLoadModal() {
            const meta = document.getElementById('load-modal-meta');
            const closeId = meta && meta.getAttribute('data-close-id');
            if (closeId) {
                const btn = document.getElementById(closeId);
                if (btn) btn.click();
            }
        }
    }

    // -----------------------------------------------------------------------
    // SimControllerBehavior: WS message handler that owns the ObjectList.
    // -----------------------------------------------------------------------
    class SimControllerBehavior extends MessageHandlerBase {
        static schema = SimControllerSchema;

        static get messageTags() {
            return [_SimTags.OBJECT_LIST, _SimTags.AVAILABLE_GAUSSIANS,
                    _SimTags.SIM_STATUS, _SimTags.PATH_COMPLETIONS, _SimTags.ERROR];
        }

        constructor(options = {}) {
            const merged = { ...defaultsFromSchema(SimControllerSchema), ...options };
            super([_SimTags.OBJECT_LIST, _SimTags.AVAILABLE_GAUSSIANS,
                   _SimTags.SIM_STATUS, _SimTags.PATH_COMPLETIONS, _SimTags.ERROR]);
            this.options = merged;
            this._objectList = new ObjectList(merged.listContainerId, this._send.bind(this));
            this._wirePathCompleter();
        }

        _conn() {
            return kaolin.core.WebSocketConnectionsManager.getOpenConnection(this.options.connectionId);
        }

        _send(tag, content) {
            const conn = this._conn();
            if (!conn) { console.warn('SimController: no open WS connection for tag=' + tag); return; }
            kaolin.io.encodeMessage(tag, content).then(enc => conn.send(enc));
        }

        onMessage(tag, content) {
            if (tag === _SimTags.OBJECT_LIST) {
                const raw = content instanceof Map ? content.get('objects') : content.objects;
                // Decoded Maps need to be converted to plain objects for ObjectList.
                const objects = (raw || []).map(o => o instanceof Map ? Object.fromEntries(o) : o);
                this._objectList.setObjects(objects);
            } else if (tag === _SimTags.AVAILABLE_GAUSSIANS) {
                const path    = content instanceof Map ? content.get('path')    : content.path;
                const entries = content instanceof Map ? content.get('entries') : content.entries;
                const entryObjs = (entries || []).map(e => e instanceof Map ? Object.fromEntries(e) : e);
                this._objectList.showGaussianSelector(path, entryObjs);
            } else if (tag === _SimTags.SIM_STATUS) {
                const msg = content instanceof Map ? content.get('message') : content.message;
                const el = document.getElementById('sim-status-display');
                if (el) el.textContent = msg || 'Idle';
            } else if (tag === _SimTags.PATH_COMPLETIONS) {
                const raw = content instanceof Map ? content.get('completions') : content.completions;
                this._showCompletions(Array.isArray(raw) ? raw : []);
            } else if (tag === _SimTags.ERROR) {
                const msg = content instanceof Map ? content.get('message') : content.message;
                const errEl = document.getElementById('load-error-msg');
                if (errEl) { errEl.textContent = msg || 'Unknown error'; }
                else { console.error('SimulateApp error:', msg); }
            }
        }

        // ---- path auto-complete ----

        _wirePathCompleter() {
            // Attach Tab-key listener on the path input, document-wide click to dismiss.
            document.addEventListener('keydown', (e) => {
                if (e.key !== 'Tab') return;
                const input = document.getElementById('load-path-input');
                if (document.activeElement !== input) return;
                e.preventDefault();
                const val = input.value || '';
                this._send(_SimTags.PATH_COMPLETE, { path: val });
            });

            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') this._hideCompletions();
            });

            document.addEventListener('click', (e) => {
                const dd = document.getElementById('path-completions-dropdown');
                const input = document.getElementById('load-path-input');
                if (dd && !dd.contains(e.target) && e.target !== input) {
                    this._hideCompletions();
                }
            });
        }

        // Set a React-controlled input's value in a way that React actually sees.
        // Directly writing input.value bypasses React's synthetic event tracking,
        // so the Dash callback still reads the old value on button click.
        _setInputValue(input, value) {
            const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            setter.call(input, value);
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }

        _showCompletions(completions) {
            const dd = document.getElementById('path-completions-dropdown');
            const input = document.getElementById('load-path-input');
            if (!dd || !input) return;

            if (completions.length === 0) {
                this._hideCompletions();
                return;
            }

            // Single unambiguous completion: fill directly, no dropdown.
            if (completions.length === 1) {
                this._setInputValue(input, completions[0]);
                this._hideCompletions();
                return;
            }

            // Fill the common prefix shared by all completions.
            const prefix = completions.reduce((a, b) => {
                let i = 0;
                while (i < a.length && i < b.length && a[i] === b[i]) i++;
                return a.slice(0, i);
            });
            if (prefix.length > (input.value || '').length) {
                this._setInputValue(input, prefix);
            }

            // Populate dropdown list.
            dd.innerHTML = '';
            completions.forEach((c) => {
                const item = document.createElement('div');
                item.className = 'px-2 py-1';
                item.style.cursor = 'pointer';
                item.style.fontSize = '0.85rem';
                item.style.fontFamily = 'monospace';
                item.textContent = c;
                item.addEventListener('mouseenter', () => { item.style.background = '#e9ecef'; });
                item.addEventListener('mouseleave', () => { item.style.background = ''; });
                item.addEventListener('mousedown', (e) => {
                    e.preventDefault();   // keep focus on input
                    this._setInputValue(input, c);
                    this._hideCompletions();
                    // If it's a directory, request another completion immediately.
                    if (c.endsWith('/')) {
                        this._send(_SimTags.PATH_COMPLETE, { path: c });
                    }
                });
                dd.appendChild(item);
            });
            dd.style.display = 'block';
        }

        _hideCompletions() {
            const dd = document.getElementById('path-completions-dropdown');
            if (dd) { dd.style.display = 'none'; dd.innerHTML = ''; }
        }
    }

    BehaviorRegister.register(
        'simulate_controller',
        SimControllerBehavior,
        'Segment-list-style object list with collapsible position/orientation sliders for the simulate app.'
    );
})();
