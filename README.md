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

![alt text](https://drive.google.com/open?id=17YxaR9WTKepwcgE2t8j2B1rQppV3I4GU)
### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
