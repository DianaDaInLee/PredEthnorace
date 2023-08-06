Step-by-Step Guidelines to Implement Hybrid Ethnoracial Prediction
================
Diana Da In Lee (<dl2860@columbia.edu>)
Last Updated: 06 August, 2023

This guideline walks through how to implement the proposed hybrid
approach to ethnoracial prediction in Lee, Diana, and Yamil Velez.
“[Measuring descriptive representation at scale: Methods for predicting
the race and ethnicity of public officials.](https://osf.io/tpsv6/)”
(2023+).

We will use the US local elections dataset compiled by [de-Benedictis
Kessner et al (2023)](https://dash.harvard.edu/handle/1/37373139). We’ve
sampled 100 elections from the main dataset with 30% of the names with
verified ethnoracial information:

    ## # A tibble: 6 × 7
    ##   state city            year firstname surname    fullname         race     
    ##   <chr> <chr>          <dbl> <chr>     <chr>      <chr>            <chr>    
    ## 1 OH    columbus        2021 shannon   hardin     shannon hardin   <NA>     
    ## 2 MA    boston          2021 michael   flaherty   michael flaherty <NA>     
    ## 3 NJ    passaic         2021 john      bartlett   john bartlett    caucasian
    ## 4 CA    orange          2021 katrina   foley      katrina foley    <NA>     
    ## 5 TX    san antonio     2021 greg      brockhouse greg brockhouse  <NA>     
    ## 6 FL    st. petersburg  2021 gina      driscoll   gina driscoll    <NA>

## 1. Image-Based Prediction

We first want to collect images associated with each name and make
predictions based on the images. We provide two python functions
`img_search` and `img_pred` to do these steps. For users who don’t have
python installed in their computer, we also provide a Google Colab
demonstration here:
[PredEthnorace_Demo.ipynb](https://github.com/DianaDaInLee/PredEthnorace/blob/main/PredEthnorace_Demo.ipynb).

### 1.1. Collect Images

Collect profile images of the individuals in the dataset, using their
first and last names as a keyword. Users may use any public engine. We
provide a python function `img_search` (in `img_pred.py`) that allows
for users to collect images from [DuckDuckGo](https://duckduckgo.com).
The function automatically 1) removes images with no faces; 2) removes
images with multiple faces; and 3) de-dupes identical faces (remove one
of the images that are detected to be of a same person as the one in
another image).

The `img_search` function in takes the following arguments:

- `fullname` (string): First and last name to be used as a keyword for
  image search.
- `maxnum` (integer): Total number of images you want to collect per
  keyword.
- `imgfolder` (string): Name of the folder you want the images to be
  saved. It will automatically create the folder in your working
  directory.

Each image will be saved in jpg as \[`keyword`\]\_\[0 through
`maxnum`\].jpg. All images associated with each keyword will be stored
in its own folder named after the keyword (so make sure to dedup the
full names in your dataset!).

``` python
img_search(fullname = 'diana lee', maxnum = 3, imgfolder = 'demo')
glob.glob('demo/*/*')
```

``` python
['demo/diana_lee/diana_lee_0.jpg',
 'demo/diana_lee/diana_lee_1.jpg',
 'demo/diana_lee/diana_lee_2.jpg']
```

Using the demo dataset,

``` python
# Python
import pyreadr
df = pyreadr.read_r('demo.rdata') 
demo = df["demo"]

# Collect Images
for index, row in demo.iterrows():
    print(row['fullname'])
    img_search(fullname = row['fullname'], maxnum = 3, imgfolder = 'demo')
```

### 1.2. Generate Predictions

#### 1.2.1 Use CNN Model

We provide a python function `img_pred` (in `img_pred.py`) that allows
for users to predict ethnorace for each image using `deepface` developed
by [Sefik Serengil](https://github.com/serengil/deepface) that uses a
pre-trained CNN model
[VGG-Face](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/)
to predict ethnorace from an image. This package is available in python
and is fairly straightforward to use.

The `img_pred` function takes the following arguments:

- `imgfolder` (string): Name of the folder where the images are saved.
  Should be the same as the folder name used in `img_search`.
- `subfolder` (string): Name of a particular subfolder (i.e., keyword)
  under `imgfolder` that you want predictions to be performed. Default
  is NULL, which will generate predictions for all images in all
  subfolders saved in `imgfolder`.
- `det` (string): face detector backend to opencv (default), retinaface,
  mtcnn, ssd, dlib, mediapipe or yolov8. See
  <https://github.com/serengil/deepface> for differences across
  different detectors.
- `out_csv` (boolean): Save the result as a csv file to a `imgfolder`.
  Default is TRUE.

``` python
pred = img_pred(imgfolder = 'demo', det = "opencv", out_csv = True)
```

    ##         asian     indian        black        white middle.eastern
    ## 1 60.16002656  0.7348647  0.096435158 2.662113e+01   1.5252286546
    ## 2 87.95337677  3.1764846  0.195944449 6.775063e-01   0.0492574181
    ## 3 98.26723337  0.1850487  0.001583018 8.093083e-01   0.0083999599
    ## 4  0.35295452  2.0282285 97.163039446 3.709444e-03   0.0037028243
    ## 5 23.88515025 16.9022009  6.558652222 9.683461e+00   9.3953385949
    ## 6  0.05296969  0.3474497 99.452441898 4.515984e-04   0.0003267264
    ##   latino.hispanic   dominant_race                                       fn
    ## 1      10.8623109           asian           demo/diana_lee/diana_lee_0.jpg
    ## 2       7.9474255           asian           demo/diana_lee/diana_lee_1.jpg
    ## 3       0.7284286           asian           demo/diana_lee/diana_lee_2.jpg
    ## 4       0.4483673           black demo/shannon_hardin/shannon_hardin_0.jpg
    ## 5      33.5752010 latino hispanic demo/shannon_hardin/shannon_hardin_1.jpg
    ## 6       0.1463617           black demo/shannon_hardin/shannon_hardin_2.jpg
    ##   detector
    ## 1   opencv
    ## 2   opencv
    ## 3   opencv
    ## 4   opencv
    ## 5   opencv
    ## 6   opencv

Not that there are many many different CNN models available. Our
function is merely a demonstration using one of these models.

#### 1.2.2 Aggregate Generated Predictions

Note that the predictions are made for *each* image. To get a single
vector of predicted probability distribution for a given name, you can
take a simple average or weighted average by the order in which the
images appear in the search engine (these are identifiable by the file
name). In our validation, we find that up-weighting images that appear
first results in a slightly higher accuracy.

``` r
pred_agg <- pred %>%
  separate(fn, c('imgfolder', 'keyword', 'filename'), sep = '/') %>%
  mutate(order = abs(as.numeric(stringr::str_extract(filename, '[0-9]+')) - 3)) %>%
  group_by(keyword) %>%
  summarize(across(asian:latino.hispanic, ~weighted.mean(.x, w = order)))
head(pred_agg)
```

    ## # A tibble: 6 × 7
    ##   keyword            asian indian   black   white middle.eastern latino.hispanic
    ##   <chr>              <dbl>  <dbl>   <dbl>   <dbl>          <dbl>           <dbl>
    ## 1 adam_mcgough     0.00146  0.243  0.0106 85.7          12.3                1.78
    ## 2 andrea_waner    18.3      0.709  0.286  72.9           2.90               4.93
    ## 3 betsy_wilkerson  6.13     1.52  90.9     0.0120        0.00440            1.46
    ## 4 bill_kinne       4.70     1.07   0.319  80.1           6.33               7.51
    ## 5 blaine_griffin   5.81     3.34  73.2     7.52          4.97               5.18
    ## 6 brenda_fincher  17.2      7.82  37.1    13.0           7.36              17.6

``` r
# Merge it with Raw Data
pred_agg <- rename_at(pred_agg, vars(asian:latino.hispanic), ~ paste0('image.', .x))
demo1 <- left_join(demo %>% mutate(keyword = tolower(gsub(" ", "_", fullname))), pred_agg)
```

    ## Joining with `by = join_by(keyword)`

## 2. Surname-Based Prediction

This part is very simple. Use `wru` package created by [Imai and Khanna
(2016)](https://imai.fas.harvard.edu/research/race.html) to generate
surname-based predictions:

``` r
library(wru)
demo2 <- predict_race(voter.file = demo1, surname.only = T)
demo2 <- demo2 %>% rename_at(vars(matches('^pred\\.')), ~ gsub('pred', 'bayes', .x))
```

## 3. Run multinomial logistic regression

``` r
library(nnet)
cov <- names(demo2)[which(grepl('bayes|image', names(demo2)))]
eq  <- as.formula(paste('race ~ ', paste(cov, collapse = ' + ')))
print(eq)
```

    ## race ~ image.asian + image.indian + image.black + image.white + 
    ##     image.middle.eastern + image.latino.hispanic + bayes.whi + 
    ##     bayes.bla + bayes.his + bayes.asi + bayes.oth

``` r
mod <- multinom(eq, data = demo2)
```

    ## # weights:  52 (36 variable)
    ## initial  value 40.202536 
    ## iter  10 value 17.107391
    ## iter  20 value 7.841235
    ## iter  30 value 4.328976
    ## iter  40 value 1.717132
    ## iter  50 value 0.001449
    ## final  value 0.000067 
    ## converged

``` r
demo2$pred_race = predict(mod, newdata = demo2, "class")
```

Users are free to employ fancier machine learning algorithms (but our
validation finds that this simple multinomial model performs just as
well).
