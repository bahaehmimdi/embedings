import torch
from transformers import T5EncoderModel, T5Tokenizer
import gradio as gr

# Load text encoder from PixArt/LTX repository stack
model_id = "PixArt-alpha/PixArt-XL-2-1024-MS"
print("Loading tokenizer and text encoder...")
tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(
    model_id, subfolder="text_encoder", torch_dtype=torch.float16
).to("cuda")
print("Model loaded successfully!")


def extract_embeddings(prompt, negative_prompt):
  max_sequence_length = 120

  # Tokenize positive prompt
  text_inputs = tokenizer(
      prompt,
      padding="max_length",
      max_length=max_sequence_length,
      truncation=True,
      return_tensors="pt",
  )
  with torch.no_grad():
    prompt_embeds = text_encoder(text_inputs.input_ids.to("cuda"))[0]

  # Tokenize negative prompt
  uncond_inputs = tokenizer(
      negative_prompt,
      padding="max_length",
      max_length=max_sequence_length,
      truncation=True,
      return_tensors="pt",
  )
  with torch.no_grad():
    negative_prompt_embeds = text_encoder(uncond_inputs.input_ids.to("cuda"))[0]

  # Save tensors locally on the space container
  output_path = "ltx_embeddings.pt"
  torch.save(
      {
          "prompt_embeds": prompt_embeds.cpu(),
          "negative_prompt_embeds": negative_prompt_embeds.cpu(),
      },
      output_path,
  )

  return output_path


# Build UI layout that doubles as an automated API endpoint
with gr.Blocks() as demo:
  gr.Markdown("# LTX Cloud Embedding Extractor")
  with gr.Row():
    prompt_input = gr.Textbox(
        label="Prompt", value="A cinematic shot of a futuristic city"
    )
    neg_input = gr.Textbox(label="Negative Prompt", value="low quality, blurry")

  output_file = gr.File(label="Download .pt Embedding File")
  submit_btn = gr.Button("Generate Embeddings")

  submit_btn.click(
      fn=extract_embeddings,
      inputs=[prompt_input, neg_input],
      outputs=output_file,
  )

if __name__ == "__main__":
  demo.launch(server_name="0.0.0.0", server_port=7860)
