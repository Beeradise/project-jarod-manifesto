# Project Jarod — Architectural Manifesto

**One Mind, Three Worlds** — the architectural manifesto for NeuroCore, the shared
HDC/LNN/VSA learning engine ported across Screeps, Timberborn, and (eventually) Minecraft.

Open `index.html` in a browser to read it, or enable GitHub Pages on this repo (Settings →
Pages → deploy from `main` branch, root) to host it at a public URL.

This is a living document — Claude is a named collaborator per its own knowledge-integrity
doctrine (§02): every claim is verified, estimated, or flagged unknown, never filled in with
plausible-sounding fluff.

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT). 
Copyright (c) 2026 Beeradise.

## Acknowledgments & Prior Art

The architecture of NeuroCore stands on the shoulders of some incredible open-source research and engineering. Specifically, this project would not be possible without the work of:

* **Ramin Hasani:** The learning engine directly utilizes code and foundational concepts from Ramin Hasani's pioneering research on Liquid Neural Networks (LNNs) at MIT CSAIL. His work on continuous-time recurrent neural networks is the backbone of how NeuroCore handles dynamic, sequential decision-making.
* **George Hotz:** While NeuroCore does not use any direct code from his repositories, the underlying philosophy of the engine—specifically the tensor calculations and the drive for minimalist, deeply optimized, from-scratch AI architectures—is heavily inspired by George Hotz and his work on `tinygrad`.
