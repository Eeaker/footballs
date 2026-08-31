# Third-party notices

## TVCalib

This module's homogeneous pixel-to-world point projection convention was
adapted from `pixel2world_homography_example.ipynb` in TVCalib:

- Source: https://github.com/MM4SPA/tvcalib
- Copyright (c) 2022 MM4SPA
- License: MIT

The complete TVCalib model was deliberately not vendored. It targets automatic
camera calibration from standard soccer-pitch line segmentation and introduces
PyTorch/Kornia/model-weight dependencies that are not required for the manual
four-point v1 workflow.

MIT License

Copyright (c) 2022 MM4SPA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The SoccerNet `sn-calibration` source was reviewed as an architectural
reference only. No source code was copied because the inspected repository did
not contain a root license file.

