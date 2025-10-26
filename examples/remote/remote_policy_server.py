import numpy as np
from fastapi import FastAPI, Request, Response

from lerobot.utils.messaging import pack_msg, unpack_msg

app = FastAPI()


@app.post("/predict")
async def predict(request: Request):
    data = await request.body()
    obs_input = unpack_msg(data)

    inf_cfg = obs_input.get("inference_config", {})
    n_action_steps = (
        inf_cfg.get("n_action_steps")
        or inf_cfg.get("n_actions")
        or inf_cfg.get("chunk_size")
        or inf_cfg.get("horizon")
        or 1
    )

    # Try to infer batch size from any array-like input
    B = None
    for v in obs_input.values():
        if isinstance(v, np.ndarray):
            if v.ndim >= 1:
                B = int(v.shape[0])
                break
    if B is None:
        # Fallback to 1 if nothing array-like found
        B = 1

    action_dim = 7  # set to your actual action dimension
    actions = np.zeros((B, n_action_steps, action_dim), dtype=np.float32)

    packed = pack_msg({"actions": actions})
    return Response(content=packed, media_type="application/octet-stream")