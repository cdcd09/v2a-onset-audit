# Licensing and attribution

Project code is distributed under the [MIT license](../../LICENSE).

## Third-party code

- **CondFoleyGen:** event-onset detector and historical NMS adaptation in
  [`code/eval/onset_metrics.py`](../../code/eval/onset_metrics.py).
  Copyright (c) 2023 Yuexi Du; [MIT notice](condfoleygen_LICENSE.txt).
  Source: https://github.com/XYPB/CondFoleyGen.
- **MMAudio:** experiment architecture in [`code/mmaudio/model/`](../../code/mmaudio/model/)
  and [`code/mmaudio/ext/`](../../code/mmaudio/ext/), with study-specific student and
  auxiliary-onset modules. Copyright (c) 2024 Sony Research;
  [MIT notice](mmaudio_LICENSE.txt). Source: https://github.com/hkchengrex/MMAudio.

Original source comments also attribute component implementations to DiT, OpenAI GLIDE,
Stability AI sd3-ref, Black Forest Labs Flux, and flow-matching. Those comments are retained.
The project license does not relicense external checkpoints, feature extractors, codecs,
vocoders, or source recordings.

## Figures and data

Author-created figures and numerical summaries in this repository are offered under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute JoonHo Byun, Taegon Kim,
Juyeon Park, and Unsang Park, *Training Refines Onset Placement but Erodes Event Correspondence*,
WACV 2027, and indicate changes. This grant covers only rights held by the authors.

The companion data package provides component-specific licenses. Its Greatest Hits
annotations originate from Andrew Owens and collaborators, *Visually Indicated Sounds*,
CVPR 2016, under the project's CC BY 4.0 terms: https://andrewowens.com/vis/.
Original video/audio recordings and third-party pretrained weights are not included in
this repository. Source-data identifiers do not transfer rights in the underlying media.
