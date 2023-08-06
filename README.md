Step-by-Step Guidelines to Implement Hybrid Ethnoracial Prediction
================
Diana Da In Lee (<dl2860@columbia.edu>)
2023-08-05

This guideline walks through how to implement the proposed hybrid
approch to ethnoracial prediction in Lee, Diana, and Yamil Velez.
“[Measuring descriptive representation at scale: Methods for predicting
the race and ethnicity of public officials.](https://osf.io/tpsv6/)”
(2023+).

We will use the following example data:

    ##    place_fips state         city year       office firstname   surname race
    ## 2     3611000    NY      Buffalo 2019 City Council   rasheed     wyatt <NA>
    ## 7     3615000    NY  Cheektowaga 2019 City Council christine  adamczyk  whi
    ## 9     3649121    NY Mount Vernon 2019 City Council      lisa  copeland <NA>
    ## 13    3650617    NY New Rochelle 2019 City Council    albert tarantino  whi
    ## 32    1714000    IL      Chicago 2019 City Council     maria    hadden  bla
    ## 34    1714000    IL      Chicago 2019 City Council     andre   vasquez  his

## 1. Image-Based Prediction

### 1.1. Collect Images

Collect profile images of the individuals in the dataset, using their
first and last names as a keyword. Users may use any public engine. We
provide a python script `collect_img.py` that allows for users to
collect images from [DuckDuckGo](https://duckduckgo.com). The function
automatically 1) removes images with no faces; 2) removes images with
multiple faces; and 3) de-dupes identical faces (remove one of the
images that are detected to be of a same person as the one in another
image).

The `search` function in `collect_img.py` takes four arguments:

- `fullname`: First and last name to be used as a keyword for image
  search.
- `maxnum`: Total number of images you want to collect per keyword.
- `imgfolder`: Name of the folder you want the images to be saved.
- `detector`: Image detector option. Choose from `opencv`, `ssd`,
  `mtcnn`, `dlib`, `retinaface`, `mediapipe`, `yunet`, and `yolov8`. For
  differences across the detectors, see
  <https://github.com/serengil/deepface>.

### 1.2. Generate Predictions

Many different CNN models are available. Here, we demonstrate our method
using `deepface` developed by [Sefik
Serengil](https://github.com/serengil/deepface) that uses a pre-trained
CNN model
[VGG-Face](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/)
to predict ethnorace from an image. This package is available in python
and is fairly straightforward to use.

See Google Colab demonstration that runs 1.1 and 1.2 in python here:.

## 2. Surname-Based Prediction

This part is very simple. Use `wru` package created by [Imai and Khanna
(2016)](https://imai.fas.harvard.edu/research/race.html) to generate
surname-based predictions:

``` r
library(wru)
demo <- predict_race(voter.file = demo, surname.only = T)
demo <- demo %>% rename_at(vars(matches('^pred')), ~ gsub('pred', 'bayes', .x))
head(demo)
```

    FALSE   place_fips state         city year       office firstname   surname race
    FALSE 1    3611000    NY      Buffalo 2019 City Council   rasheed     wyatt <NA>
    FALSE 2    3615000    NY  Cheektowaga 2019 City Council christine  adamczyk  whi
    FALSE 3    3649121    NY Mount Vernon 2019 City Council      lisa  copeland <NA>
    FALSE 4    3650617    NY New Rochelle 2019 City Council    albert tarantino  whi
    FALSE 5    1714000    IL      Chicago 2019 City Council     maria    hadden  bla
    FALSE 6    1714000    IL      Chicago 2019 City Council     andre   vasquez  his
    FALSE    bayes.whi   bayes.bla  bayes.his   bayes.asi  bayes.oth
    FALSE 1 0.71364817 0.189217795 0.02937698 0.005880129 0.06187693
    FALSE 2 0.95283797 0.002621187 0.01913765 0.002067164 0.02333603
    FALSE 3 0.61355541 0.283934405 0.02693715 0.006351567 0.06922147
    FALSE 4 0.90709588 0.003516968 0.05106396 0.005893921 0.03242927
    FALSE 5 0.81136274 0.108702568 0.01814122 0.010346342 0.05144713
    FALSE 6 0.04061328 0.004410540 0.93606698 0.008712616 0.01019659

## 3. Run multinomial logistic regression

``` r
library(nnet)
#mod <- multinom(race ~ img.whi + img.bla + img.his + img.asi + bayes.whi + bayes.bla + bayes.his + bayes.asi + bayes.oth, data = demo)
#demo$pred_race = predict(mod, newdata = demo, "class")
```

Users are free to employ fancier machine learning algorithms (but our
validation finds that this simple multinomial model performs just as
well).
