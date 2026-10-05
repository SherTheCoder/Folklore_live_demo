# Utho Kikkar: Keyword Spotting for Indian Languages

Made by Team Folklore for SIH 2026. A static page for trying version 2 of the Utho Kikkar wake word model (the 64-channel `alt_64ch` build) with your microphone. Say ਉੱਠੋ ਕਿੱਕਰ / उठो किक्कर and the kikkar tree blooms. Everything runs in the page and no audio leaves the browser.

This model is a prototype. The final one will be built at SIH 2026.

## How close is it to the device

- Features match `pymicro-features` (the same C frontend) exactly.
- Model outputs match TFLite Micro's reference kernels exactly. The terminal tester uses LiteRT, which rounds differently in a few places, so its outputs can differ from this page's by a step now and then.
- Chrome and Safari resample the microphone to 16 kHz themselves. Firefox won't, so the worklet does it with the same filter the terminal tester uses.
