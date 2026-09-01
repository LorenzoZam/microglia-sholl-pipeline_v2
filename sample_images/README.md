# Bundled Iba1 examples

These four images are single-channel, 8-bit Iba1 TIFFs prepared from the
public BioImage Archive study
[S-BIAD1280](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1280).
They cover the four exposure/region folders represented in the source data.

The examples were extracted from channel 2 (one-based indexing), identified in
the source metadata as Iba1 / Alexa Fluor 488. TIFF resolution tags correspond
to approximately **0.454546 µm/px** in both axes. The Streamlit demo pre-fills
this value for bundled examples; users must verify calibration for uploads.

| Bundled file | Source composite | Extracted TIFF SHA-256 |
|---|---|---|
| `Exposed_CRBLM_F03_Iba1.tif` | `Exposed CRBLM/F image 03.tif` | `1140ed4279e7911b96fc0adae84b271afc569214685d01c610fa4073e212f743` |
| `Exposed_STN_M03_Iba1.tif` | `Exposed STN/M image 03.tif` | `18000608409fab176e8ec8dd9df85580fc8565b40b51cbf5c6d4c86694ec274f` |
| `Sham_CRBLM_F05_Iba1.tif` | `Sham CRBLM/F image 05.tif` | `f8aed2204c331fe5ff4d29f08d38429d39eb15f435c4d3b8caee1788e4331592` |
| `Sham_STN_M03_Iba1.tif` | `Sham STN/M image 03.tif` | `ed9564219e28b899b8e968342417d1832a6c86dafe65a30078ef54e7e8f197b5` |

Staining protocol:
[protocols.io DOI 10.17504/protocols.io.kqdg3xbbeg25/v1](https://doi.org/10.17504/protocols.io.kqdg3xbbeg25/v1).

The images are demonstration inputs, not a validation benchmark.
