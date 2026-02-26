"""The assembled model: frozen vision tower -> projection -> LoRA decoder.

The composition is the whole point of the project, so the wiring is explicit:

1. the vision encoder is frozen and produces pooled features;
2. the projection head maps those features into the text embedding width;
3. the same projected vector is passed to the decoder as cross-attention
   memory, so captions are actually conditioned on the image;
4. only the projection and the LoRA adapters receive gradients.
"""
