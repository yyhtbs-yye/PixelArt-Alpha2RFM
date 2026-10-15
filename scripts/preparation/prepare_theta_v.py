"""Prepare frozen conditioning and strictly map a Theta RAW checkpoint to native PixArt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'output/training_dependencies'))
sys.path.insert(0, str(ROOT / 'output/t5_dependencies'))

import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    torch.set_num_threads(4)
    source, work, snapshot = Path(args.source).resolve(), Path(args.output), Path(args.snapshot).resolve()
    work.mkdir(parents=True, exist_ok=True)
    entries = json.loads(Path(args.prompts).read_text())
    if len(entries) != 4:
        raise ValueError('Exactly four validation prompts are required')
    config = json.loads((source / 'transformer/config.json').read_text())
    expected = dict(num_layers=28, num_attention_heads=16, attention_head_dim=72, in_channels=4,
                    out_channels=8, patch_size=2, sample_size=64, qk_norm='rms_norm', qk_norm_eps=1e-6,
                    use_additional_conditions=False, noise_preconditioning_config=None)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Unexpected source architecture: {key}')
    schedule = json.loads((source / 'scheduler/scheduler_config.json').read_text())
    for key, value in dict(prediction_type='epsilon', beta_schedule='linear', beta_start=.0001,
                           beta_end=.02, num_train_timesteps=1000, rescale_betas_zero_snr=False).items():
        if schedule.get(key) != value:
            raise ValueError(f'Unexpected source schedule: {key}')
    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(snapshot / 'tokenizer', local_files_only=True)
    encoder = T5EncoderModel.from_pretrained(snapshot / 'text_encoder', local_files_only=True,
                                            dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
    prompts = [entry['prompt'] for entry in entries]
    tokens = tokenizer([''] + [prompt.strip() for prompt in prompts], max_length=120,
                       padding='max_length', truncation=True, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        features = encoder(**tokens).last_hidden_state.cpu()[:, None]
    masks = tokens.attention_mask.bool().cpu()
    torch.save(dict(empty=features[:1], empty_mask=masks[:1], condition=features[1:], mask=masks[1:],
                    prompts=prompts, seeds=[entry['seed'] for entry in entries],
                    t5_snapshot=str(snapshot), preprocessing='strip only; matches source TextConditioner'),
               work / 'conditioning.pt')
    del encoder, tokens
    torch.cuda.empty_cache()
    from diffusion.model.builder import build_model
    from diffusion.utils.theta_import import import_theta_state, make_initial_metadata
    model = build_model('PixArt_XL_2', input_size=64, pred_sigma=True, model_max_length=120,
                        class_dropout_prob=0., qk_norm='rms_norm', use_grad_checkpoint=False, lewei_scale=1.).float()
    weight_path = source / 'transformer/diffusion_pytorch_model.safetensors'
    weights = load_file(weight_path)
    import_theta_state(model, weights, features[0, 0])
    provenance = dict(source=str(source), source_weights='raw', source_prediction_type='epsilon',
                      source_sha256=hashlib.file_digest(weight_path.open('rb'), 'sha256').hexdigest(),
                      source_tensor_count=len(weights), source_optimizer_step=10000,
                      mapping='all learned tensors strictly mapped; regenerate sin/cos position buffer; encode empty text',
                      conditioning_sha256=hashlib.file_digest((work / 'conditioning.pt').open('rb'), 'sha256').hexdigest())
    native_config = dict(prediction_type='epsilon', qk_norm='rms_norm', qk_norm_eps=1e-6,
                         source_model=str(source), source_weights='raw', conditioning_mode='empty_text')
    metadata = make_initial_metadata(native_config, .18215, model)
    torch.save(dict(state_dict=model.state_dict(), metadata=metadata, import_provenance=provenance),
               work / 'initial_raw_epsilon.pth')
    (work / 'import.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (work / 'prompts.json').write_text(json.dumps(entries, indent=2) + '\n')
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
