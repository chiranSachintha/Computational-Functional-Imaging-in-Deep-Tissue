# de-blurring-technique-with-GAN
**final-year-project(Collaborative project with Harvard University, Cambridge, USA)**

## ABSTRACT
**Keywords**: Fluorescence imaging, dynamic images, two-photon imaging, temporal
focusing
Capturing fast biological dynamic events through scattering biological tissue for a large
imaging field of view (FoV) is a challenging task. State of the art two-photon point
scanning microscopy, which is the workhorse in deep tissue imaging experiments, is
speed limited; usually, FoV is sacrificed in the favor of temporal-resolution or vice
versa. For rapid wide-field two-photon imaging previously demonstrated a large FoV
two-photon microscopy technique, called DEEP-TFM, based on temporal focusing and
coded illumination.
However, to reconstruct one large-FoV image frame, multiple coded illuminations are
needed, restricting temporal resolution. However, in many applications, temporal
dynamics appear as spatially constraint signal, on a slow varying (near-static)
background signal. In this work, we demonstrate a deep neural network (DNN)
framework to separate the dynamic signal from the static signal at the true imaging
framerate. We first estimate the sum of the static and dynamic images using a deep
neural network. Then we feed this output, along with one illumination code and its
corresponding scattered image, to a second-deep neural network to estimate the current
dynamic image.
We demonstrate our framework on simulated images and experimental images.
One Paragraph of project description goes here

## INTRODUCTION

### Introduction to two-photon imaging

Fluorescence microscopy is an imaging technique where the whole sample is illuminated
with light of a specific wavelength, exciting fluorescent molecules within it.

▪ Dynamic Imaging: Any diagnostic image that varies with time. Also called realtime imaging<br />
▪ Static Imaging: any diagnostic image that is fixed or frozen in time.

Initially, one photon is used for exciting fluorescent molecules. But the emission light is
focused along the entire illuminated cone, not just at the focus. Moreover, the following
constraints occur with one photon (conventional) fluorescence imaging when observing
thick specimens.

▪ Physical limitations: Objective lens working distance is limited<br />
▪ Tissue penetration depth: due to absorption and scattering of the emission light

To overcome this constraint two-photon imaging is introduced. ICapt can restrict excitation
to a tiny focal volume which allows the visualization of living tissue at depths unachievable
with one photon fluorescence microscopy.
There are several microscopes that use this two photon imaging technique.
When using point scanning emission photons from the focal point scattered or not being
collected by a point detector and assign to a single pixel of the image. Hence this method
is very slow. Moreover, Point scanning two-photon imaging is not fast enough to see large
regions in the brain at functional speed. 

The wide-field two photon microscopy is a good alternative for this. In it a short-pulse
laser beam is scattered by a grating. To maintain the widefield imaging the emission light
from the focal plane is typically recorded by a camera. This may assign incorrect pixels on
the detector, resulting in degradation of both resolution and signal to noise. Although
temporal focusing light can deliver light through scattering media, collecting light back is
not much efficient in this modality due to the scattering. Hence Temporal Focusing
Microscope (TFM) shows background haze at shallower image planes.
Escobet et al [1] have proposed a single pixel detector while modulating excitation light in
the imaged field of view by projecting a set of patterns. The modulated images are then
recorded using the microscope.

As they use single-pixel detection the acquisition time is quite high and depends on the
field of view. They require the same number of illumination patterns as the number of pixels
in the imaged field of view. Thus, no evident speed up over Point Scanning Two-Photon
Microscopy (PSTPM) was demonstrated. 

![scattered image](Images/2.png)

In Professor Peter So’s group in Massachusetts Institute of Technology (MIT), they have
developed a computational imaging technique to see a large field of views with two-photon
microscopy. They are also using a set of patterns, but the emission light is detected by a
camera. The camera is used to obtain a large field of view at the detector.
In practice, TFM images are minimally affected by scattering at or near the surface; as the
imaging depth increases, scattering gradually degrades only the high-frequency
information in the images.
DEEP-TFM combines the information about the excitation patterns with the acquired
images, to computationally reconstruct a de-scattered image.

Experimentally, to de-scatter a single FoV, multiple patterned excitations (and images) are
needed; the number depends on the loss of high-frequency information due to scattering,
and hence on the imaging depth.

That group has developed a mathematical model for reconstructing images using the
patterns and the acquired images in each image plane. However, the existing model is timeconsuming and not suitable for dynamic image reconstruction.
Objectives of our project:

1. Simulation of a synthetic dataset.
2. One frame reconstruction for static images.
3. Multiple image frame reconstruction for dynamic images.

### Main Novelty of the project:

DEEP-TFM is a novel computational wide-field technology for deep tissue multiphoton
microscopy. DEEP-TFM can resolve images with similar quality to point scanning twophoton microscopy for static images with the existing reconstruction algorithm. That
approach of DEEP-TFM is FoV independent. Our goal is the obtain multiple frames for
dynamic images obtained from DEEP-TFM. This is a novel concept as these microscopes
are mainly used for capturing static images. For living tissues, it is hard to obtain a static
image. By implementing a model which can reconstruct multiple frames for dynamic
images, it enables to capture fast biological dynamic events through scattering biological
tissue for a large imaging FoV. 

### METHODOLOGY

Our approach can be mainly divided into three parts as we mentioned in the objectives.

1. Simulation of synthetic dataset
2. Developing a model for one frame reconstruction for the static images.
3. Developing a model for multi-frame reconstruction for the dynamic images.

![](synthetic%20dataset.PNG)

