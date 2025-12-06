DAVIS 2017 Unsupervised dataset: The 2019 DAVIS Challenge on Video Object Segmentation
============================================================================================

Package containing the `DAVIS 2017` dataset with the unsupervised annotations, as part of *The 2019 DAVIS Challenge on Video Object Segmentation*.
More information on the [challenge website](http://davischallenge.org/challenge2019/index.html).

Documentation
-----------------

The directory is structured as follows:

 * `ROOT/JPEGImages`: Set of input video sequences provided in the form of JPEG images.
   Video sequences are available at Full-Resolution (4k, 1440p, 1080p, etc.) and 480p resolution.

 * `ROOT/Annotations_unsupervised`: Set of manually annotated multiple-object reference segmentations
   for the foreground objects. Annotations are available at Full-Resolution and 480p resolution.

 * `ROOT/ImageSets`: Files containing the set of sequences on each of the dataset subsets.

Credits
---------------
All sequences if not stated differently are owned by the authors of `DAVIS` and are
licensed under Creative Commons Attributions 4.0 License, see [Terms of Use].

See SOURCES.md for the online sources of all videos. Please refer to the provided links for their terms-of-use. Videos missing in SOURCES.md were collected by the authors.

Citation
--------------

Please cite `DAVIS 2019`, `DAVIS 2017`, and `DAVIS 2016` in your publications if they help your research:
    `@article{Caelles_arXiv_2019,
      author    = {Sergi Caelles and
                   Jordi Pont-Tuset and
                   Federico Perazzi and
                   Alberto Montes and
                   Kevis-Kokitsi Maninis and
                   Luc {Van Gool}},
      title     = {The 2019 DAVIS Challenge on Video Object Segmentation},
      journal   = {arXiv},
      year      = {2019}
    }`


    `@article{Pont-Tuset_arXiv_2017,
      author    = {Jordi Pont-Tuset and
                   Federico Perazzi and
                   Sergi Caelles and
                   Pablo Arbel\'aez and
                   Alexander Sorkine-Hornung and
                   Luc {Van Gool}},
      title     = {The 2017 DAVIS Challenge on Video Object Segmentation},
      journal   = {arXiv:1704.00675},
      year      = {2017}
    }`


    `@inproceedings{Perazzi_CVPR_2016,
      author    = {Federico Perazzi and
                   Jordi Pont-Tuset and
                   Brian McWilliams and
                   Luc {Van Gool} and
                   Markus Gross and
                   Alexander Sorkine-Hornung},
      title     = {A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year      = {2016}
    }`

Terms of Use
--------------

`DAVIS 2017 Unsupervised` is released under the Creative Commons License:
  [CC BY-NC](http://creativecommons.org/licenses/by-nc/4.0).

In synthesis, users of the data are free to:

1. **Share** - copy and redistribute the material in any medium or format.
2. **Adapt** - remix, transform, and build upon the material.

The licensor cannot revoke these freedoms as long as you follow the license terms.

Contacts
------------------
- [Jordi Pont-Tuset](http://jponttuset.cat)
- [Federico Perazzi](https://graphics.ethz.ch/~perazzif)
- [Sergi Caelles](https://sergicaelles.com/)

