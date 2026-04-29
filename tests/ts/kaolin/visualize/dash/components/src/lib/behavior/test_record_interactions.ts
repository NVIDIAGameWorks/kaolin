// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

import { assert } from 'chai';

import {
    InteractionHandlerName,
    InteractionEventHandlerName,
    RecordInteractionsBehavior,
    eventToSerializable,
    eventFromSerializable,
    replayInteractions,
} from '@kaolin/lib/behavior/record_interactions';
import { registerDom, unregisterDom } from '@test/helpers/dom';

const H = InteractionHandlerName;

// A fixed, non-square target rect; left/top are non-zero so coordinates exercise
// the viewport-relative offset (clientX - rect.left, etc.).
const RECT = { left: 100, top: 50, width: 200, height: 400 };

/** A minimal element whose only relevant feature is a fixed bounding rect. */
function rectEl(): HTMLElement {
    return {
        getBoundingClientRect: () => ({
            ...RECT, right: RECT.left + RECT.width, bottom: RECT.top + RECT.height,
            x: RECT.left, y: RECT.top, toJSON: () => ({}),
        }),
    } as unknown as HTMLElement;
}

/** Build a synthetic pointer/mouse event of `type` with randomized fields, inside {@link RECT}. */
function makeRandomEvent(type: string): any {
    return {
        type,
        clientX: RECT.left + Math.random() * RECT.width,
        clientY: RECT.top + Math.random() * RECT.height,
        button: Math.floor(Math.random() * 3),
        buttons: 1,
        ctrlKey: Math.random() < 0.5, shiftKey: Math.random() < 0.5,
        altKey: Math.random() < 0.5, metaKey: Math.random() < 0.5,
        detail: Math.floor(Math.random() * 3),
        pointerId: Math.floor(Math.random() * 100),
        pointerType: type.startsWith('pointer') ? 'mouse' : 'mouse',
        pressure: Math.random(),
        isPrimary: true,
        timeStamp: Math.floor(Math.random() * 100000),
        currentTarget: rectEl(),
    };
}

// One recorded step: a pointer/click handler + its event, or a setOption call.
type Interaction =
    | { handler: InteractionEventHandlerName; event: any }
    | { handler: InteractionHandlerName.SetOption; name: string; value: any };

// The interaction stream applied to every behavior: each pointer/click handler
// twice, in a jumbled order, interleaved with two setOption calls. Built once so
// all behaviors (and the replay target) see identical input.
const INTERACTIONS: Interaction[] = [
    { handler: H.SetOption, name: 'color', value: '#abc' },
    { handler: H.PointerEnter, event: makeRandomEvent('pointerenter') },
    { handler: H.PointerDown, event: makeRandomEvent('pointerdown') },
    { handler: H.PointerMove, event: makeRandomEvent('pointermove') },
    { handler: H.Click, event: makeRandomEvent('click') },
    { handler: H.PointerUp, event: makeRandomEvent('pointerup') },
    { handler: H.DoubleClick, event: makeRandomEvent('dblclick') },
    { handler: H.PointerMove, event: makeRandomEvent('pointermove') },
    { handler: H.PointerCancel, event: makeRandomEvent('pointercancel') },
    { handler: H.SetOption, name: 'thickness', value: 4 },
    { handler: H.PointerDown, event: makeRandomEvent('pointerdown') },
    { handler: H.PointerUp, event: makeRandomEvent('pointerup') },
    { handler: H.PointerLeave, event: makeRandomEvent('pointerleave') },
    { handler: H.Click, event: makeRandomEvent('click') },
    { handler: H.DoubleClick, event: makeRandomEvent('dblclick') },
    { handler: H.PointerEnter, event: makeRandomEvent('pointerenter') },
    { handler: H.PointerLeave, event: makeRandomEvent('pointerleave') },
    { handler: H.PointerCancel, event: makeRandomEvent('pointercancel') },
];

/** Feed one interaction to a behavior (a pointer/click handler call or a setOption call). */
function applyInteraction(behavior: RecordInteractionsBehavior, step: Interaction): void {
    if ('event' in step) {
        (behavior as any)[step.handler](step.event);
    } else {
        behavior.setOption(step.name, step.value);
    }
}

/** Tally records (or interactions) by their `handler` tag. */
function countByHandler(items: Array<{ handler: string }>): Record<string, number> {
    const counts: Record<string, number> = {};
    for (const item of items) {
        counts[item.handler] = (counts[item.handler] ?? 0) + 1;
    }
    return counts;
}

// Metadata fields an event and its SerializableEvent carry verbatim.
const META_FIELDS = ['button', 'buttons', 'ctrlKey', 'shiftKey', 'altKey', 'metaKey',
    'detail', 'pointerId', 'pointerType', 'pressure', 'isPrimary', 'timeStamp'];

describe('visualize/dash/components/src/lib/behavior/test_record_interactions.ts', () => {

    // eventFromSerializable mints real PointerEvent/MouseEvent objects (its
    // nativeEvent), which need a browser DOM.
    before(() => { registerDom(); });
    after(async () => { await unregisterDom(); });

    describe('eventToSerializable - eventFromSerializable round-trip and correctness', () => {
        it('serializes every field, then reconstructs a faithful event', () => {
            const event = makeRandomEvent('pointerdown');
            const s = eventToSerializable(event, rectEl(), 'canvas');

            assert.equal(s.type, event.type, 'type carried through');
            assert.closeTo(s.fracX, (event.clientX - RECT.left) / RECT.width, 1e-9, 'fracX = (clientX - left) / width');
            assert.closeTo(s.fracY, (event.clientY - RECT.top) / RECT.height, 1e-9, 'fracY = (clientY - top) / height');
            assert.equal(s.targetWidth, RECT.width, 'targetWidth from rect');
            assert.equal(s.targetHeight, RECT.height, 'targetHeight from rect');
            assert.equal(s.targetRole, 'canvas', 'targetRole label stored');
            for (const f of META_FIELDS) {
                assert.deepEqual((s as any)[f], event[f], `serialized: ${f}`);
            }

            // Reconstruct against the same frame -> client coords recover exactly.
            const e = eventFromSerializable(s, rectEl());
            assert.equal(e.type, event.type, 'reconstructed type');
            assert.closeTo(e.clientX, event.clientX, 1e-9, 'clientX recovered from fraction + same frame');
            assert.closeTo(e.clientY, event.clientY, 1e-9, 'clientY recovered from fraction + same frame');
            assert.equal(e.pageX, e.clientX, 'pageX mirrors clientX');
            assert.equal(e.pageY, e.clientY, 'pageY mirrors clientY');
            for (const f of META_FIELDS) {
                assert.deepEqual((e as any)[f], event[f], `reconstructed: ${f}`);
            }
            assert.equal(e.nativeEvent.type, event.type, 'nativeEvent carries the DOM type');
            assert.instanceOf(e.nativeEvent, PointerEvent, 'pointer types mint a PointerEvent');
        });

        it('is frame-independent and mints a MouseEvent for non-pointer types', () => {
            const s = eventToSerializable(makeRandomEvent('click'), rectEl());
            // With no element, coordinates resolve against targetWidth/Height at the origin.
            const e = eventFromSerializable(s, null);
            assert.closeTo(e.clientX, s.fracX * RECT.width, 1e-9, 'no frame -> reconstruct from targetWidth at origin');
            assert.closeTo(e.clientY, s.fracY * RECT.height, 1e-9, 'no frame -> reconstruct from targetHeight at origin');
            assert.instanceOf(e.nativeEvent, MouseEvent, 'non-pointer types mint a MouseEvent');
        });
    });

    describe('RecordInteractionsBehavior records correctly', () => {
        it('captures only the handlers selected by the record option', () => {
            const all = new RecordInteractionsBehavior({ record: 'all' });
            const pointerOnly = new RecordInteractionsBehavior({ record: [H.PointerDown, H.PointerUp] });
            const clickAndOption = new RecordInteractionsBehavior({ record: [H.Click, H.SetOption] });

            for (const behavior of [all, pointerOnly, clickAndOption]) {
                for (const step of INTERACTIONS) {
                    applyInteraction(behavior, step);
                }
            }

            // 'all' records the whole stream verbatim, one record per interaction.
            assert.equal(all.getRecords().length, INTERACTIONS.length, 'all: one record per interaction');
            assert.deepEqual(countByHandler(all.getRecords()), countByHandler(INTERACTIONS),
                'all: per-handler counts mirror the input stream');

            // Subsets keep only their selected handlers, twice each; setOption only
            // when selected.
            assert.deepEqual(countByHandler(pointerOnly.getRecords()),
                { [H.PointerDown]: 2, [H.PointerUp]: 2 },
                'pointerOnly: only onPointerDown / onPointerUp, twice each');
            assert.deepEqual(countByHandler(clickAndOption.getRecords()),
                { [H.Click]: 2, [H.SetOption]: 2 },
                'clickAndOption: only onClick / setOption, twice each');
        });
    });

    describe('RecordInteractionsBehavior record then replay', () => {
        it('replays a recorded stream into an identical log on a clean behavior', () => {
            // Bind both to equally-sized frames so fractions round-trip exactly.
            const source = new RecordInteractionsBehavior({ record: 'all' });
            source.setActiveElement(rectEl());
            for (const step of INTERACTIONS) {
                applyInteraction(source, step);
            }
            const recorded = source.getRecords();

            const replayed = new RecordInteractionsBehavior({ record: 'all' });
            replayed.setActiveElement(rectEl());
            replayInteractions(replayed, recorded);

            assert.deepEqual(replayed.getRecords(), recorded,
                'replaying the recorded stream reproduces an identical log');
        });
    });
});
