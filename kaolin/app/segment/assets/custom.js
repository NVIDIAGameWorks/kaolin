
// Mirror of MessageTags in kaolin/app/segment/handler.py: the wire tags this
// client exchanges with the server. Keep in sync with the server enum.
const MessageTags = Object.freeze({
    // Incoming requests the server handles
    RENDER: 'render',
    SET_CAMERA: 'set_camera',
    SAM_SEGMENT: 'sam_segment',
    PROJECT: 'mask_project',
    AGGREGATE: 'aggregate',
    SEGMENT: 'segment',
    SEGMENT_REFRESH: 'segment_refresh',
    // Outgoing from the server, received here
    SET_SEGMENTS: 'set_segments',
    DONE_WITH_MASK: 'done_with_mask',
});


// Fire a tagged message at the open websocket connection.
async function send_server_request(messageTag, content = {}) {
    console.log(`Sending server request: ${messageTag}`);
    const encoded = await kaolin.io.encodeMessage(messageTag, content);
    kaolin.core.WebSocketConnectionsManager.getOpenConnection()?.send(encoded);
}


// Grab the current mask (alpha channel of the mask layer canvas) and ask the
// server to project it onto the 3D selection using the given action.
async function send_mask_project_request(mask_canvas_id, action) {
    const mask = await kaolin.util.canvas.blobAlphaChannelFromCanvas(mask_canvas_id);
    await send_server_request(MessageTags.PROJECT, {
        "mask": mask,
        "action": action,
    });
}


function prepareMode(mode) {
    kaolin.core.event.requestBehaviorActiveStatus("{send_cam_id}", true);
}


// Hybrid behavior: SVG annotation + send-message. On every click, draws a
// positive/negative marker on the SVG layer and forwards the *full* current
// set of clicks over the configured WebSocket connection.
(function () {
    const { SvgAnnotationBehavior } = kaolin.lib.behavior;
    const { getRelativeCoordinates } = kaolin.util.cursor;
    const { BehaviorRegister, defaultsFromSchema } = kaolin.core.behavior;
    const { WebSocketConnectionsManager } = kaolin.core;
    const { encodeMessage } = kaolin.io;

    const DEFAULT_ASSETS = {
        positive: {
            elements: [
                { name: 'circle', options: {
                    cx: '0', cy: '0', r: '11.206225680933851',
                    fill: '#22aa22', stroke: 'white',
                    'stroke-width': '2.801556420233463',
                    'pointer-events': 'visiblePainted',
                }},
                { name: 'line', options: {
                    x1: '-5.6031128404669', y1: '0',
                    x2: '5.6031128404669',  y2: '0',
                    stroke: 'white', 'stroke-width': '2.801556420233463',
                }},
                { name: 'line', options: {
                    x1: '0', y1: '-5.6031128404669',
                    x2: '0', y2: '5.6031128404669',
                    stroke: 'white', 'stroke-width': '2.801556420233463',
                }},
            ],
            scale: 0.1,
        },
        negative: {
            elements: [
                { name: 'circle', options: {
                    cx: '0', cy: '0', r: '11.206225680933851',
                    fill: '#cc2222', stroke: 'white',
                    'stroke-width': '2.801556420233463',
                    'pointer-events': 'visiblePainted',
                }},
                { name: 'line', options: {
                    x1: '-5.6031128404669', y1: '0',
                    x2: '5.6031128404669',  y2: '0',
                    stroke: 'white', 'stroke-width': '2.801556420233463',
                }},
            ],
            scale: 0.1,
        },
    };

    const SamPointSchema = {
        isPositive: {
            kind: 'bool',
            default: true,
            description: 'Whether subsequent clicks are positive (foreground) or negative (background) prompts.',
            uiBound: true,
        },
        connectionId: {
            kind: 'string',
            default: '',
            description: 'Id of the WebSocket connection (registered via add_websocket_connection).',
            uiBound: false,
        },
        messageTag: {
            kind: 'string',
            default: MessageTags.SAM_SEGMENT,
            description: 'Tag prepended to each outgoing point-set message.',
            uiBound: false,
        },
    };

    class SamPointBehavior extends SvgAnnotationBehavior {
        static schema = SamPointSchema;

        constructor(options = {}) {
            // The schema is the single source of truth for SAM-specific option
            // defaults; merge caller-supplied overrides on top of it.
            const merged = { ...defaultsFromSchema(SamPointBehavior.schema), ...options };
            super({
                assets: DEFAULT_ASSETS,
                activeAsset: merged.isPositive ? 'positive' : 'negative',
                onClickAction: 'add_or_remove',
            });

            this.options = {
                ...this.options,
                ...merged,
            };

            this.points = [];
        }

        updateForOptions(name, value) {
            if (name === 'isPositive') {
                this.options.activeAsset = value ? 'positive' : 'negative';
            }
        }

        onMarkerAdded(marker, index, event) {
            const { fracX, fracY } = getRelativeCoordinates(event, this.element);
            this.points.splice(index, 0, { x: fracX, y: fracY, label: this.options.isPositive ? 1 : 0 });
            this._sendPoints();
        }

        onMarkerRemoved(marker, index, event) {
            this.points.splice(index, 1);
            this._sendPoints();
        }

        reset(options) {
            super.reset(options);
            this.points = [];
        }

        async _sendPoints() {
            const { connectionId, messageTag } = this.options;
            const labels = this.points.map(pt => pt.label);
            const positions = this.points.map(pt => [pt.x, pt.y]);
            const encoded = await encodeMessage(messageTag, { labels: labels, positions: positions });
            WebSocketConnectionsManager.getOpenConnection(connectionId)?.send(encoded);
        }
    }

    BehaviorRegister.register(
        'sam_point',
        SamPointBehavior,
        'Click to add positive/negative SAM point prompts; forwards the full point set on every click.'
    );
})();



// Mirror of SegmentAction in kaolin/app/segment/handler.py. There is no
// deselect action server-side: deselect is a SELECT carrying `value: false`.
const SegmentAction = Object.freeze({
    ADD: 'add',
    DELETE: 'delete',
    UPDATE: 'update',
    SHOW: 'show',
    SELECT: 'select',
});

(function () {
    const { defaultsFromSchema, BehaviorRegister, MessageHandlerBase } = kaolin.core.behavior;

    const SegControllerSchema = {
        listContainerId: {
            kind: 'string',
            description: 'Id of the DOM container the segment list is rendered into.',
            uiBound: false,
        },
        switchToModeWhenDoneWithMask: {
            kind: 'string',
            description: 'Mode to switch to when done with mask.',
            uiBound: false,
        }
    };

    class SegControllerBehavior extends MessageHandlerBase {
        static schema = SegControllerSchema;

        constructor(options = {}) {
            // The schema is the single source of truth for option defaults;
            // merge caller-supplied overrides on top of it.
            const merged = { ...defaultsFromSchema(SegControllerBehavior.schema), ...options };
            super([MessageTags.SET_SEGMENTS, MessageTags.DONE_WITH_MASK]);
            this.options = merged;
            this.segList = new SegmentList(this.options.listContainerId);
            this._registerEvents();
        }

        onMessage(messageTag, messageContent) {
            if (messageTag === MessageTags.SET_SEGMENTS) {
                const segments = messageContent?.get('segments') ?? [];
                this.segList.setSegments(segments);
            } else if (messageTag === MessageTags.DONE_WITH_MASK) {
                // The server consumed the 2D mask, so reset the drawing/point
                // behaviors that produced it back to their initial state.
                if (this.options.switchToModeWhenDoneWithMask) {
                    kaolin.core.event.requestMode(this.options.switchToModeWhenDoneWithMask);
                } 
            }
        }

        _containerEl() {
            return this.options.listContainerId
                ? document.getElementById(this.options.listContainerId)
                : null;
        }

        // All DOM events for the rendered list are owned here, not in SegmentList,
        // so the list stays a pure renderer. Listeners are delegated on `document`
        // and scoped to our container: they survive every re-render and tolerate
        // the container not yet existing at construction time.
        _registerEvents() {
            document.addEventListener('click', (event) => {
                const container = this._containerEl();
                if (!container) {
                    return;
                }
                // Action buttons take priority; clicks inside an input are ignored.
                const btn = event.target.closest?.('[data-action]');
                if (btn && container.contains(btn)) {
                    const index = btn.dataset.index !== undefined ? Number(btn.dataset.index) : null;
                    const value = btn.dataset.value !== undefined ? btn.dataset.value === 'true' : undefined;
                    this._handleAction(btn.dataset.action, index, value);
                    return;
                }
                if (event.target.closest?.('input')) {
                    return;
                }
                // The whole row is the click target. Activating the add row opens
                // its name field (and focuses it); activating a segment row reveals
                // its extra controls. Invisible segments are rejected by setActive.
                const addRow = event.target.closest?.('[data-role="add-row"]');
                if (addRow && container.contains(addRow)) {
                    this.segList.setActive(SegmentList.ADD_KEY);
                    container.querySelector('[data-role="add-name"]')?.focus();
                    return;
                }
                const entry = event.target.closest?.('[data-index]');
                if (entry && container.contains(entry)) {
                    this.segList.setActive(Number(entry.dataset.index));
                }
            });

            // Inline rename is available only on the active row.
            document.addEventListener('dblclick', (event) => {
                const nameEl = event.target.closest?.('[data-role="seg-name"]');
                const container = this._containerEl();
                if (!nameEl || !container || !container.contains(nameEl)) {
                    return;
                }
                const entry = nameEl.closest('[data-index]');
                if (entry && this.segList.activeKey === Number(entry.dataset.index)) {
                    this._beginEditName(nameEl, Number(entry.dataset.index));
                }
            });

            // Enter submits / Escape cancels the add-segment name field.
            document.addEventListener('keydown', (event) => {
                const addInput = event.target.closest?.('[data-role="add-name"]');
                const container = this._containerEl();
                if (!addInput || !container || !container.contains(addInput)) {
                    return;
                }
                if (event.key === 'Enter') {
                    event.preventDefault();
                    this._submitAdd();
                } else if (event.key === 'Escape') {
                    event.preventDefault();
                    this.segList.setActive(null);
                }
            });
        }

        // Dispatch a segment action to the server. The server
        // (`_execute_segment_task`) keys every op on `name`, dispatches on
        // `action`, and reads a per-action payload (`value`, `new_mask`,
        // `new_name`). All actions ride the single `segment` message tag. Note
        // the server has no DESELECT: it is a SELECT carrying `value: false`.
        _handleAction(action, index, value) {
            if (action === SegmentAction.ADD) {
                this._submitAdd();
                return;
            }
            const item = (index !== null && !Number.isNaN(index)) ? this.segList.items[index] : null;
            if (!item) {
                return;
            }
            const request = { name: item.name, action };
            switch (action) {
                case SegmentAction.SHOW:
                    // Optimistically flip visibility for immediate feedback and
                    // send the new desired state (server reads `value`).
                    // item.visible = !item.visible;
                    //this.segList.update_list();
                    request.value = !item.visible;
                    break;
                case SegmentAction.SELECT:
                    // Select (value=true) adds this segment's mask to the 3D
                    // selection; deselect (value=false) removes it. The server
                    // folds both into SELECT keyed on the boolean `value`.
                    request.value = value !== false;
                    break;
                case SegmentAction.UPDATE:
                    // Re-bake this segment's mask from the current 3D selection:
                    // the server reads the `new_mask` flag and calls
                    // `_update_segment_mask(name)`.
                    request.new_mask = true;
                    break;
                case SegmentAction.DELETE:
                    // No extra payload; the server deletes by `name`.
                    break;
                default:
                    return;
            }
            send_server_request(MessageTags.SEGMENT, request);
        }

        // Read the add-segment name field, notify the server, then collapse the
        // add row back to its prompt.
        _submitAdd() {
            const container = this._containerEl();
            const input = container?.querySelector('[data-role="add-name"]');
            const name = input ? input.value.trim() : '';
            // Server `_add_segment(name)` bakes the new segment from the current
            // 3D selection; only the name needs to travel with the action.
            const request = { name, action: SegmentAction.ADD };
            send_server_request(MessageTags.SEGMENT, request);
            this.segList.setActive(null);
        }

        // Swap the name label for an inline input; commit on blur/Enter, cancel on
        // Escape. Double-click maps to double-tap on touch devices.
        _beginEditName(nameEl, index) {
            const item = this.segList.items[index];
            if (!item) {
                return;
            }
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'form-control form-control-sm flex-grow-1';
            input.style.minWidth = '0';
            input.value = item.name ?? '';
            nameEl.replaceWith(input);
            input.focus();
            input.select();

            let settled = false;
            const commit = () => {
                if (settled) return;
                settled = true;
                const newName = input.value.trim();
                const oldName = item.name;
                // Optimistic local update; the server's segment broadcast reconciles.
                item.name = newName;
                this.segList.update_list();
                // Rename is an UPDATE keyed on the current (server) name, carrying
                // the desired `new_name` (server calls _update_segment_name).
                if (newName && newName !== oldName) {
                    send_server_request(MessageTags.SEGMENT,
                        { name: oldName, action: SegmentAction.UPDATE, new_name: newName });
                }
            };
            input.addEventListener('blur', commit);
            input.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    input.blur();
                } else if (event.key === 'Escape') {
                    event.preventDefault();
                    settled = true;
                    input.removeEventListener('blur', commit);
                    this.segList.update_list();
                }
            });
        }
    }

    BehaviorRegister.register(
        'seg_controller',
        SegControllerBehavior,
        'Manages client-side segmentation app state');
})();


// ---------------------------------------------------------------------------
// SegmentInfo: typed description of a single segment shown in a SegmentList.
//
// `FIELDS` is the single source of truth for the accepted keys and their
// defaults. The constructor reads each field generically from the source —
// which may be a plain object or a decoded server message `Map` — so adding a
// field here is the only change needed to wire it end to end (no per-field
// hand-wiring at call sites).
// ---------------------------------------------------------------------------
class SegmentInfo {
    static FIELDS = {
        name: '',
        visible: true,
        info: '',
        can_delete: true,
    };

    constructor(source = {}) {
        const get = (key) => (source instanceof Map ? source.get(key) : source[key]);
        for (const [field, fallback] of Object.entries(SegmentInfo.FIELDS)) {
            const value = get(field);
            this[field] = value ?? fallback;
        }
    }
}


// ---------------------------------------------------------------------------
// SegmentList: renders a list of segment items into a container element.
// Items are coerced to `SegmentInfo` (see its `FIELDS` for the accepted shape).
// Call `update_list()` to (re)render the Bootstrap-styled list.
// ---------------------------------------------------------------------------
class SegmentList {
    static ADD_KEY = 'add';

    constructor(elementId) {
        this.elementId = elementId;
        this.items = [];
        // The single active row: a segment index, SegmentList.ADD_KEY for the
        // add-segment row, or null when nothing is active ("one or fewer").
        this.activeKey = null;
    }

    get element() {
        return document.getElementById(this.elementId);
    }

    // Replace the tracked items and re-render the list. Accepts raw plain
    // objects or decoded server `Map`s; each is coerced to a `SegmentInfo`.
    // (`update_list` prunes a now-invalid active segment.)
    setSegments(items) {
        this.items = (items ?? []).map((item) => new SegmentInfo(item));
        this.update_list();
    }

    // Activate a single row: a segment index, ADD_KEY, or null to clear. A
    // segment can be activated only while visible (invisible rows must be made
    // visible first); invalid/invisible targets are ignored so they do not
    // clear an existing selection. Idempotent: re-activating the current row is
    // a no-op (no re-render), which keeps inline editing stable.
    setActive(key) {
        let next;
        if (key === null || key === SegmentList.ADD_KEY) {
            next = key;
        } else if (Number.isInteger(key) && key >= 0 && key < this.items.length
                   && this.items[key].visible) {
            next = key;
        } else {
            return;
        }
        if (next === this.activeKey) {
            return;
        }
        this.activeKey = next;
        this.update_list();
    }

    // Build a Bootstrap icon button. Tooltips use the native `title` attribute
    // (also surfaced as `aria-label`) so they need no Bootstrap JS and remain
    // usable on touch devices. `flex-shrink-0` keeps the tap target intact when
    // the row is narrow; `fs-5` makes the glyphs comfortably large.
    _iconButton({ icon, title, action, index, value, colorClass = 'text-body', disabled = false }) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `btn btn-link p-1 lh-1 fs-4 flex-shrink-0 text-decoration-none ${colorClass}`;
        btn.title = title;
        btn.setAttribute('aria-label', title);
        if (action) {
            btn.dataset.action = action;
        }
        if (index !== undefined) {
            btn.dataset.index = String(index);
        }
        if (value !== undefined) {
            btn.dataset.value = String(value);
        }
        btn.disabled = disabled;
        btn.innerHTML = `<i class="bi ${icon}"></i>`;
        return btn;
    }

    // Rebuild the HTML list from the current items as a Bootstrap list-group.
    update_list() {
        const container = this.element;
        if (!container) {
            return;
        }

        // Prune an active segment that no longer exists or is no longer visible;
        // the ADD_KEY row is always available.
        if (typeof this.activeKey === 'number'
            && !(this.activeKey >= 0 && this.activeKey < this.items.length
                 && this.items[this.activeKey].visible)) {
            this.activeKey = null;
        }

        const list = document.createElement('ul');
        // `kaolin-seg-container` (styles.css) controls the shared single borders
        // between items and the tiny info subtitle.
        list.className = 'list-group kaolin-seg-container';

        // Add-segment row at the top. Clicking the row activates it; while active
        // it shows a name field + add button instead of the prompt.
        const addItem = document.createElement('li');
        addItem.className = 'list-group-item';
        addItem.dataset.role = 'add-row';
        if (this.activeKey === SegmentList.ADD_KEY) {
            addItem.classList.add('kaolin-seg-active');
            const row = document.createElement('div');
            row.className = 'd-flex align-items-center gap-2';

            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'form-control form-control-sm flex-grow-1';
            input.style.minWidth = '0';
            input.placeholder = 'Segment name';
            input.dataset.role = 'add-name';
            row.appendChild(input);

            row.appendChild(this._iconButton({
                icon: 'bi-plus-lg', title: 'Add segment',
                action: SegmentAction.ADD, colorClass: 'text-success',
            }));
            addItem.appendChild(row);
        } else {
            const row = document.createElement('div');
            row.className = 'd-flex align-items-center justify-content-center gap-2 text-success fw-semibold';
            row.innerHTML = '<i class="bi bi-plus-lg"></i><span>Add Segment</span>';
            addItem.appendChild(row);
        }
        list.appendChild(addItem);

        this.items.forEach((item, index) => {
            const entry = document.createElement('li');
            entry.className = 'list-group-item';
            entry.dataset.index = String(index);
            if (this.activeKey === index) {
                entry.classList.add('kaolin-seg-active');
            }

            // Row 1 (always shown): name + tiny info subtitle on the left, the
            // visibility toggle on the right. `min-width: 0` lets `text-truncate`
            // ellipsize; full text is preserved in `title`.
            const nameRow = document.createElement('div');
            nameRow.className = 'd-flex justify-content-between align-items-center gap-2';

            const labels = document.createElement('div');
            labels.className = 'flex-grow-1';
            labels.style.minWidth = '0';

            const name = document.createElement('div');
            name.className = 'fw-semibold text-truncate';
            name.textContent = item.name ?? '';
            name.title = item.name ?? '';
            name.dataset.role = 'seg-name';
            labels.appendChild(name);

            // Always rendered (even when empty) so the layout is stable.
            const info = document.createElement('div');
            info.className = 'kaolin-seg-info text-muted text-truncate';
            info.textContent = item.info ?? '';
            info.title = item.info ?? '';
            labels.appendChild(info);

            nameRow.appendChild(labels);

            nameRow.appendChild(this._iconButton({
                icon: item.visible ? 'bi-eye' : 'bi-eye-slash',
                title: item.visible ? 'Hide' : 'Show',
                action: SegmentAction.SHOW, index,
                colorClass: item.visible ? 'text-body' : 'text-secondary',
            }));
            entry.appendChild(nameRow);

            // Row 2 (extra controls): shown only for the single active row.
            if (this.activeKey === index) {
                const actionRow = document.createElement('div');
                actionRow.className = 'd-flex justify-content-end align-items-center gap-1 mt-2';

                actionRow.appendChild(this._iconButton({
                    icon: 'bi-union', title: 'Select segment',
                    action: SegmentAction.SELECT, index, value: true, colorClass: 'text-secondary',
                }));
                actionRow.appendChild(this._iconButton({
                    icon: 'bi-subtract', title: 'Deselect segment',
                    action: SegmentAction.SELECT, index, value: false, colorClass: 'text-secondary',
                }));
                actionRow.appendChild(this._iconButton({
                    icon: 'bi-box-arrow-in-left', title: 'Update to current selection (will remove from other segments)',
                    action: SegmentAction.UPDATE, index, colorClass: 'text-secondary',
                }));
                if (item.can_delete) {
                    actionRow.appendChild(this._iconButton({
                        icon: 'bi-trash', title: 'Delete',
                        action: SegmentAction.DELETE, index, colorClass: 'text-danger',
                    }));
                }

                entry.appendChild(actionRow);
            }

            list.appendChild(entry);
        });

        container.replaceChildren(list);
    }
}