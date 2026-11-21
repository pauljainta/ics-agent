# Research animation sources

The 2:30 film presents a conceptual IT → OT → physical-process storyline in authorized research testbeds. It is not captured exploitation footage. Its IT foothold and cross-boundary access are assumed narrative conditions, not a demonstrated capability claim for ICS-Agent. The historic incidents were malware campaigns, not AI-agent attacks, and the animation does not reproduce their malware.

- **Stuxnet (discovered 2010):** [ENISA, Stuxnet Analysis](https://www.enisa.europa.eu/news/enisa-news/stuxnet-analysis). Supports the description of Siemens industrial-control software and industrial-system targeting. [Ralph Langner, To Kill a Centrifuge (2013), primary technical analysis hosted by Yale](https://cs.yale.edu/homes/jf/Langner.pdf) supports the description of centrifuge-control manipulation; the film does not assert a damage count or reproduce the attack.
- **Industroyer (December 2016 incident; analysis published 2017):** [ESET Research, Industroyer retrospective](https://www.welivesecurity.com/2022/06/13/industroyer-cyber-weapon-brought-down-power-grid/). Supports the description of industrial-protocol use, substation equipment, and the Kyiv power outage.
- **TRITON (2017):** [UK NCSC, TRITON Malware Targeting Safety Controllers](https://www.ncsc.gov.uk/information/triton-malware-targeting-safety-controllers). Supports the description of Triconex safety-controller targeting. See also [CISA, HatMan safety-system malware analysis](https://www.cisa.gov/sites/default/files/documents/MAR-17-352-01%20HatMan%E2%80%94Safety%20System%20Targeted%20Malware_S508C.pdf).

The displayed process curves, thresholds, network topology, agent activity, and timing are illustrative animation data. They are not experimental results or details of an operational facility. The video uses original code-drawn visuals and an original synthesized ambient soundtrack, with no third-party image, video, or audio assets.

## Reproduce

Use Python with `Pillow` and `imageio-ffmpeg` installed, then run `python render_overview.py` in this directory. An existing `ffmpeg` on PATH can be used instead. `python render_overview.py --preview` produces just the poster/contact sheet. The renderer uses standard DejaVu fonts from the system, or a local `fonts/` folder, with a Pillow fallback.

Deliverables: `ics-agent-research-overview.mp4` (150 seconds), `overview-preview.gif`, `overview-poster.png`, `overview.srt`, and `storyboard.json`. Captions are burned into the film; the SRT is also provided for accessibility. The soundtrack contains no narration.
