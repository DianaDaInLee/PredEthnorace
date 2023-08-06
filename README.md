Guidelines
================
Diana Da In Lee
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
provide a demo python code `collect_img.py` that allows for users to
collect images from [DuckDuckGo](https://duckduckgo.com).

The function automatically 1) removes images with no faces; 2) removes
images with multiple faces; and 3) de-dupes identical faces (remove one
of the images that are detected to be of a same person as the one in
another image).

The `search` function in `collect_img.py` takes four arguments:

- `fullname`: First and last name to be used as a keyword for image
  search.
- `code`: Suffix of the jpeg filename to be used. Files will be saved
  with the naming convention “\[code\]\_\[1 through maxnum\].jpg”
- `maxnum`: Total number of images you want to collect per keyword.
- `imgfolder`: Name of the folder you want the images to be saved.
- `detector`: Image prediction detector. Choose from `opencv`, `ssd`,
  `mtcnn`, `dlib`, `retinaface`, `mediapipe`, `yunet`, and `yolov8`. For
  differences across the detector, see
  <https://github.com/serengil/deepface>.

### 1.2. Generate Predictions

Many different CNN models are available. Here, we demonstrate our method
using `deepface` (<https://github.com/serengil/deepface>), which is
implemented in Python.

## 2. Surname-Based Prediction

This part is very simple. Use `wru` package created by [Imai and Khanna
(2016)](https://imai.fas.harvard.edu/research/race.html) to generate
surname-based predictions:

``` r
library(wru)
demo <- predict_race(voter.file = demo, surname.only = T)
```

    ## Proceeding with last name predictions...

    ## ℹ All local files already up-to-date!

    ## 9 (8.9%) individuals' last names were not matched.

``` r
demo <- demo %>% rename_at(vars(matches('^pred')), ~ gsub('pred', 'bayes', .x))
```

## Run multinomial logistic regression

``` r
library(nnet)
#mod <- multinom(race ~ img.whi + img.bla + img.his + img.asi + bayes.whi + bayes.bla + bayes.his + bayes.asi + bayes.oth, data = demo)
#demo$pred_race = predict(mod, newdata = demo, "class")
```

Users are free to employ fancier machine learning algorithms (but our
validation finds that this simple multinomial model performs just as
well).
