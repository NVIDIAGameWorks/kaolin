 

export enum ViewerCustomEvent {
    EDIT_BEHAVIOR ='edit_behavior',
    SET_BEHAVIOR_ACTIVE = 'set_behavior_active',
    SET_MODE = 'set_mode'
}; 

export function customViewerEventName(event: ViewerCustomEvent, viewerId: string = 'kaolin-viewer') {
    return `${event}-${viewerId}`;
}

export interface EditBehaviorCommand {
    behaviorId: string;
    setterName: string;
    value: any;
}

/**
 * Request a behavior edit by dispatching a custom event.
 * 
 * This function allows external components (e.g., Dash callbacks, UI controls) to
 * modify a behavior's properties at runtime by dispatching a custom DOM event.
 * The target viewer listens for these events and updates the specified behavior
 * accordingly.
 * 
 * @param viewerId - The unique identifier of the KaolinViewer instance
 * @param behaviorId - The identifier of the behavior to edit
 * @param setterName - The name of the setter method or property to update
 * @param value - The new value to set
 * 
 * 
 * The behavior must implement the specified setter method to handle the update.
 * 
 */
export function requestBehaviorEdit(behaviorId: string, setterName: string, value: any, viewerId: string = 'kaolin-viewer') {
    const messageType: ViewerCustomEvent = ViewerCustomEvent.EDIT_BEHAVIOR;

    const eventName = customViewerEventName(messageType, viewerId);
   
    const eventData = { 
        detail: {
            behaviorId: behaviorId,
            setterName: setterName,
            value: value
        }
    }

    const event = new CustomEvent(eventName, eventData);
    globalThis.dispatchEvent(event);
}

export function parseEditBehaviorEvent(event: CustomEvent) : EditBehaviorCommand | null {
    const command = event.detail as EditBehaviorCommand;
    return command;
} 

export const ACTIVE_BEHAVIOR_SETTER_NAME = 'setBehaviorActiveStatus';
export function requestBehaviorActiveStatus(behaviorId: string, value: boolean, viewerId: string = 'kaolin-viewer') {
    requestBehaviorEdit(behaviorId, ACTIVE_BEHAVIOR_SETTER_NAME, value, viewerId);
}

/**
 * Request a generic, schema-driven option change on a behavior.
 *
 * Sends an EDIT_BEHAVIOR command whose `setterName` is `setOption` and whose
 * `value` is `{ name, value }`. The viewer's edit dispatcher unpacks the two
 * arguments and calls `behavior.setOption(name, value)` — which itself routes
 * through the behavior's static schema (if any) for validation/casting.
 *
 * This is the preferred way to change behavior options at runtime; it replaces
 * per-option setters like `setColor` / `setThickness` / `setMode`.
 */
export const SET_OPTION_SETTER_NAME = 'setOption';
export function requestBehaviorSetOption(behaviorId: string, optionName: string, optionValue: any,
                                 viewerId: string = 'kaolin-viewer') {
    requestBehaviorEdit(behaviorId, SET_OPTION_SETTER_NAME,
                        { name: optionName, value: optionValue }, viewerId);
}


export const BEHAVIOR_RESET_SETTER_NAME = 'reset';
export function requestBehaviorReset(behaviorId: string, value: any = undefined, viewerId: string = 'kaolin-viewer') {
    requestBehaviorEdit(behaviorId, BEHAVIOR_RESET_SETTER_NAME, value, viewerId);
}


export interface SetModeCommand {
    modeName: string;
}

/**
 * Request the viewer switch to the named behavior mode.
 *
 * The viewer iterates over every registered behavior and activates it iff it appears
 * in the target mode's id set; all other behaviors are deactivated. Behavior modes
 * are declared at viewer-build time via `ViewerBuilder.add_mode(...)`.
 *
 * @param modeName - Name of the mode to activate (must have been registered via add_mode)
 * @param viewerId - The unique identifier of the KaolinViewer instance
 */
export function requestMode(modeName: string, viewerId: string = 'kaolin-viewer') {
    const eventName = customViewerEventName(ViewerCustomEvent.SET_MODE, viewerId);
    const event = new CustomEvent(eventName, { detail: { modeName } });
    globalThis.dispatchEvent(event);
}

export function parseSetModeEvent(event: CustomEvent): SetModeCommand | null {
    return event.detail as SetModeCommand;
}

