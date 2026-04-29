// Custom client-side helpers for the splat_inpaint app.
//
// `send_server_request` is the simple "fire a tagged message at the open
// websocket" helper used by buttons in main.py. The two specialized helpers
// below it package canvas pixel data (`send_opt_request` reads RGB and the
// mask, `send_inpaint_request` also reads the prompt textarea) and ship them
// over the same connection.
//
// The server side must list the messageTag in `accepted_message_tags()` for
// the dispatch to succeed (see handlers.py).


async function send_server_request(messageTag, content = {}) {
    console.log(`Sending server request: ${messageTag}`);
    const encoded = await kaolin.io.encodeMessage(messageTag, content);
    kaolin.core.WebSocketConnectionsManager.getOpenConnection()?.send(encoded);
}


function set_brush_mode(mode, ...behavior_ids) {
    // mode should be 'erase' or 'draw'
    for (const behavior_id of behavior_ids) {
        kaolin.core.event.requestBehaviorSetOption(behavior_id, "mode", mode);
    }
}


async function send_opt_request(inpaint_canvas_id, mask_canvas_id) {
    // Getting Raw data (much larger message)
    //const image = kaolin.util.canvas.typedArrayFromCanvas(inpaint_canvas_id);
    //const mask = kaolin.util.canvas.alphaChannelFromCanvas(mask_canvas_id);

    // Getting encoded data (smaller)
    const [image, mask] = await Promise.all([
        kaolin.util.canvas.blobFromCanvas(inpaint_canvas_id),
        kaolin.util.canvas.blobAlphaChannelFromCanvas(mask_canvas_id),
    ]);
    await send_server_request('train_gaussians', {
        "img": image,
        "mask": mask,
    });
}


async function send_inpaint_request(render_canvas_id, mask_canvas_id, prompt_id) {
    // Getting Raw data (much larger message)
    // const image = kaolin.util.canvas.typedArrayFromCanvas(render_canvas_id);
    // const mask = kaolin.util.canvas.alphaChannelFromCanvas(mask_canvas_id);

    const [image, mask] = await Promise.all([
        kaolin.util.canvas.blobFromCanvas(render_canvas_id),
        kaolin.util.canvas.blobAlphaChannelFromCanvas(mask_canvas_id),
    ]);

    const promptElement = document.getElementById(prompt_id);
    const prompt = promptElement ? promptElement.value : '';

    await send_server_request('inpaint', {
        "img": image,
        "mask": mask,
        "prompt": prompt,
        "neg_prompt": "bad anatomy, deformed, ugly, disfigured, cartoon, pop-art",
    });
}
