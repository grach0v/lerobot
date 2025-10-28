import numpy as np
import torch
from fastapi import FastAPI, Request, Response

from lerobot.utils.messaging import pack_msg, unpack_msg

app = FastAPI()


@app.post("/predict")
async def predict(request: Request):
    data = await request.body()
    obs_input = unpack_msg(data)

    inf_cfg = obs_input.get("inference_config", {})
    dataset_info = obs_input.get("dataset_info", {})
    n_action_steps = inf_cfg.get("n_action_steps", 10)
    action_dim = dataset_info.get("action_dof", 7)

    # Try to infer batch size from any array-like input
    batch_size = None
    for v in obs_input.values():
        if isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim > 0:
            batch_size = int(v.shape[0])
            break

    if batch_size is None:
        batch_size = 1  # Default to batch size 1 if no array-like inputs found

    actions = torch.zeros((batch_size, n_action_steps, action_dim), dtype=torch.float32)

    packed = pack_msg(actions)
    return Response(content=packed, media_type="application/octet-stream")
