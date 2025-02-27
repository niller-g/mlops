# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [X] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [x] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1

> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

43

### Question 2

> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

204424, s204470, s204510

### Question 3

> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We added Great Expectations to handle data validation, which wasn’t in the course materials. It checks that our clean_text column exists and isn’t empty, and that our dataset has enough rows for proper training. If something’s off—like missing text—it throws an error before training starts. This saved us time debugging weird data issues and let us trust the data going into our model. Integrating Great Expectations felt like a natural step toward a more robust MLOps pipeline, because it stops training automatically if any data quality checks fail, rather than letting us discover the problem later when it’s harder to fix.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used `pyproject.toml` for managing our dependencies, including `requirements.txt` and `requirements_dev.txt`. This approach allowed us to define and categorize dependencies and allow us to potentially make more levels of dependencies for example `requirements_deploy.txt`, `requirements_test.txt` etc... To get an exact copy of our development environment, a new team member would need to follow these steps:

1. Download and install Python.
2. Create a virtual environment using `.venv` by running `python -m venv .venv`.
3. Activate the virtual environment:
   * On Windows: `.venv\Scripts\activate`
   * On macOS/Linux: `source .venv/bin/activate`
4. Install the dependencies using the command: `pip install .[dev]`.

This process ensures a consistent development environment with all necessary dependencies, including those for development and testing.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

From the cookiecutter template, we used its default folder structure, including the `data`, `models` and `train` modules. We also filled out configuration files for environment and dependency management. We kept the `tests` folder from the template to organize our unit tests instead of including them along side the source code itself instead of having them right next to each file. This structure allowed us to quickly integrate Docker, CI/CD, and other MLOps practices without any refactoring. We have some .gitignore´ed folders that will be generated such as `outputs`, `models`, `data`. Some of these folders will be generated when training others when pulling data from dvc.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We used Ruff as a linting step in our CI pipeline to enforce style conventions. We also have a pre-commit hook that automatically runs Ruff to format our code, which keeps everything consistent across the team. For typing, we added type hints in the main data, model, and training functions so it’s clearer what each function expects and returns. This is especially helpful when the codebase grows and new members join. In large projects, consistent formatting and code quality checks prevent "style drift" and reduce merge conflicts. Typing also reduces bugs by catching mismatches early, and clear documentation helps everyone quickly understand how each function should be used.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We have around eight tests in total. Four focus on the data processing part—checking that our dataset downloads correctly, verifies the presence of columns like clean_text, and handles edge cases. Two tests check model creation, ensuring we can initialize DistilGPT2Model without errors and that forward passes work on dummy input. The remaining two test our training flow, confirming we can run a short training loop without crashing and produce valid outputs. We prioritized these areas because data loading and model initialization are the most critical.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Our coverage hovers around 70%. We’re focusing heavily on data utilities and core model logic, so the uncovered lines are mostly edge cases or some Docker/CLI code. Even if we reached 100% coverage, that wouldn’t guarantee a bug-free codebase—coverage only tells us whether each line was executed, not whether we tested all logical pathways or performance edge cases. Real-world issues (like concurrency, unexpected data drift, or misconfiguration) can still pop up even if every line gets run in tests. That said, pushing coverage higher does reduce the risk of regressions. Our goal is to strike a balance between thorough testing of important parts (like data transformations and model forward passes) without letting coverage targets overshadow actual usage scenarios.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We ended up no using separate branches and pull requests in this project, mainly because we were often pair-programming or making quick, non-overlapping contributions. In a typical software engineering scenario, we’d create feature branches and open PRs for every change, but here we found it simpler to commit directly to `main` and resolve small merge conflicts as they arose. This approach worked for us because we maintained close communication and generally knew which files each person was modifying. But we do recognize the value of branching and PRs in larger teams or when changes are more substantial. In those cases, feature branches help isolate new functionality, and pull requests give people a chance to review and catch mistakes before merging.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

Yes, we used DVC to version-control our processed dataset. Each time we changed our preprocessing, we committed the updated dataset with dvc add and pushed it to remote storage in the Google Cloud. That way, the code commit and the data version are aligned. If a bug was introduced by new data transformations, we had the ability to roll back both code and data. It also kept large files out of our Git repo, preventing ballooning repository sizes. Although we didn't use it much, having DVC track each dataset version is nice to have, so we can reproduce old experiments exactly.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have one main GitHub Actions workflow called ci.yml. It runs on every push and pull request, covering the following steps:

1. We run Ruff across the code to enforce style consistency and to check if anything is unformatted.

2. We run pytest on both our data tests and model tests, with coverage generation. We use the pytest-cov plugin to measure coverage.

3. Our workflow tests Python 3.11 on Ubuntu, MacOs and Windows.

4. We enable pip caching in the GitHub Actions environment so repeated runs are faster. This shortens installation times significantly, especially with packages like PyTorch and Transformers.
We also have a separate workflow for building Docker images whenever we push changes to main, which triggers Cloud Build.

Additionally when changes to the code, model or data arise, we then run integration tests with the newest data and model pulled from the Google Cloud storage. This made it possible for os to test the API with the actual data and model we would use in production. We could not run this in the CI pipeline because of the need for to pull the data from the Google Cloud storage which made it simpler to just trigger it and then check the results in Google Cloud.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used Hydra to load YAML configs in our configs folder. For instance, train.py has a Hydra decorator that reads parameters like batch_size, lr, and max_epochs. To run an experiment, we can do something like:
python src/mlops/train.py train.batch_size=8 train.lr=2e-5 train.max_epochs=3
Hydra then overrides the default values in configs/config.yaml with those command-line arguments. This keeps our hyperparameters organized in one place and makes it simple to run multiple experiments with different settings, all while logging them in W&B.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

First, we store all hyperparameters and paths in Hydra config files so we have a record of exactly how each experiment was run. Second, every run logs metrics, hyperparams, and artifacts to Weights & Biases. That gives us a versioned history of each experiment, including training curves, final model weights, and any config overrides. Third, we use DVC to version the data, so we can always roll back to the dataset used in a given run. Together, we make sure that we can replicate any experiment - we just check out the right Git commit, pull the matching dataset with DVC, and re-run the code using the same config.

--- question 13 fill here ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

We configured a Weights & Biases (W&B) sweep that automatically sampled various hyperparameters (such as learning rate, batch size, weight decay, etc.) and launched multiple training runs. In theory, this would help us discover an optimal combination without manually trying each setting. But for some reason, these sweep runs turned out to be unexpectedly slow. Even after reducing the dataset size and lowering the number of steps per epoch, we still found each run took longer than anticipated to reach a useful point in training.
As a result, we did not manage to complete a real hyperparameter search - instead, we took a small subset of our dataset - just enough to confirm that the W&B sweep setup was working. That gave us at least some initial data to upload to W&B, which you can see below. The chart is admittedly not very informative. If we wanted truly meaningful sweep results, we would probably need more computational resources and a more optimized approach, and possibly further profiling to pinpoint what was slowing down our CPU runs. ![Small-sweep-results](figures/failed-sweeps.png)

We did manage to get a bunch of useful results from training sessions and they can be seen in the following figures:
![Evaluation-results](figures/wandb_eval.png), ![Training-results](figures/wandb_train.png).

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We primarily containerized our FastAPI application using a file named `api_local.dockerfile` in the `dockerfiles/` folder. That Dockerfile sets up the environment, installs dependencies, and exposes port `8000`. Our **Docker Compose** file (**`docker-compose.yml`**) then orchestrates multiple containers at once: the API container, Prometheus for metrics scraping, Grafana for visualization, and Locust for load testing.
When we run: docker compose up --build it builds the API image locally and spins up every service on a shared network, so Prometheus can scrape the API at api:8000/metrics, and Grafana can pull data from Prometheus. This lets all teammates launch an identical environment with the same Python libraries and OS packages.
We don’t currently use a dedicated container for training, so our main focus has been deploying and monitoring the inference API in a reproducible manner. See our [Dockerfile](https://github.com/your-username/your-repo/blob/main/dockerfiles/api_local.dockerfile)

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

We used a mix of debugging methods. However, we mostly relied the debugger, pdb, to set breakpoints, step through code and inspect variables. We also used Weights & Biases to log metrics and artifacts during training, which helped us spot issues like exploding gradients or slow convergence. We didn’t do a full profiling run, but we did use the Python cProfile module to measure the time spent in different functions. This helped us gain confidence in identifying bottlenecks. We also used the `line_profiler` package to profile specific functions and lines of code, which helped us optimize our data loading and model inference steps.

However we did not feel like it was necessary to keep those parts since the application would have to grow in size before there is great benefits.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used Google Cloud Storage (GCS) for storing large datasets and model checkpoints via DVC, Google Cloud Build to automatically build Docker images when we push to main, and Secret Manager to store the Weights & Biases API key (retrieved in train.py). GCS replaced local storage for data files, Cloud Build integrated with our GitHub repo, and Secret Manager let us avoid hardcoding sensitive credentials.

Also we use Google Artifact Registry to store our Docker images and Google Cloud Run to run our FastAPI application. Artifact Registry is a private Docker registry that stores our container images, Compute Engine provides virtual machines to run our API.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We created one simple Compute Engine instance with Ubuntu. We did not specify any custom hardware, since our application is not very demanding and the training could be done in a reasonable time.

This instance was used in the early stages of development to host our FastAPI application, which was containerized in a Docker image. The Compute Engine instance was a convenient way to test our application in a cloud environment without the complexity of a full Kubernetes cluster. We could easily SSH into the instance to check logs, run tests, and monitor performance.

However, we then moved to Google Cloud Run to run our FastAPI application, which is a fully managed serverless platform that automatically scales up and down based on traffic. This was a more cost-effective solution for our use case, since we only pay for the resources we use.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![Google Cloud Storage Bucket](figures/buckets.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![Google Artifact Registry](figures/artifact-registry.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![Google Cloud Build History](figures/build-history.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We trained our model in Google Cloud using a Dockerized trainer, but did not rely on Vertex AI. Instead, we used a Compute Engine instance where we pulled our Docker image from the Artifact Registry and ran the training code inside the container. This setup gave us direct control over the environment while still leveraging the scalability of the cloud. It also allowed us to confirm that container-based training works in a real cloud environment without incurring extra complexity from Vertex AI. Our proof-of-concept was successful on the Compute Engine, and we found this approach sufficient for our current scope.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

Yes, we built a FastAPI application (api.py) that loads our fine-tuned DistilGPT2 model. We set up a /infer endpoint, so users can post a JSON or form parameter prompt, and the model returns generated text. We also added /metrics using Prometheus Python client, which lets Prometheus scrape inference latencies, CPU usage, etc. We containerized the API in api_local.dockerfile, making it easy to spin up with Docker Compose. We also have the background thread **_start_system_metrics_collection** that collects system metrics (RAM, CPU, GPU memory) for each request. This makes us able track performance in real time, which is useful for debugging and general monitoring.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We only deployed the API **locally** using Docker Compose. After building our container image from **`api_local.dockerfile`**. So we spin up the FastAPI container plus Prometheus, Grafana, and Locust on a shared Docker network. The API listens on port 8000, Prometheus scrapes /metrics, and Grafana visualizes the collected metrics. Locust runs load tests against `http://api:8000/infer` to measure inference performance under different traffic scenarios. For our current project scope, this local setup is enough for demonstration and development.

We did mange to deploy the API in the cloud using Google Cloud Run. We built a new Docker image named `api` and pushed it to Google Artifact Registry. Then we created a new service in Cloud Run, specifying the image and port. We did manage to send curl requests to the deployed service, but we also used Locust to send load tests to the API.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

For unit testing, we rely on pytest with a `test_api.py` file to ensure the `"/infer"` endpoint returns a valid response, and we mock the underlying model to avoid loading large files every time. We also conducted a basic load test with Locust, using a script (`locust_tests.py`) that spawns a small number of simulated users (2–5). Each user selects random prompts and occasionally sets a `max_length` to see if our API can handle multiple requests concurrently. We didn’t push for high concurrency or specifically tune response times, since we’re running on CPU and mostly wanted to confirm the API wouldn’t crash under a modest load. Even under those limited conditions, requests completed in a reasonable time, which is fine for our current demo.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Yes, we set up a local monitoring stack with Prometheus and Grafana. Our FastAPI app exposes a /metrics route via the prometheus_client library, which includes custom counters and gauges for training steps, validation loss, and inference latency. Prometheus scrapes this endpoint every five seconds, then stores the time-series metrics. Grafana reads from Prometheus, letting us visualize CPU usage, GPU memory, average inference time, and so on. We also tested the system under load with Locust, watching real-time dashboards in Grafana to see if latency spiked.

Here is a screenshot of the Grafana dashboard while using Locust to send some load:
Grafana-dashboard-1: ![Grafana-dashboard](figures/Grafana.png)
Grafana-dashboard-2: ![Grafana-dashboard](figures/Grafana-2.png)
Locust-dashboard: ![Locust-dashboard](figures/Locust.png)
Prometheus-cpu-usage-dashboard: ![Prometheus-dashboard](figures/Prometheus.png)

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

We used about 5 credits in total, with the most expensive service being Google Cloud Run. This was because we ran a few load tests with Locust to simulate multiple users hitting the API at once. The cost of Cloud Run is based on the number of requests and the amount of CPU and memory used, so running a large number of requests can quickly add up.

The second most expensive service was Google Artifact Registry, which we used to store our Docker images. The cost of Artifact Registry is based on the amount of storage used and the number of requests made to the registry.

Working in the cloud was a good experience overall. It allowed us to easily scale our application, deploy new versions quickly, and monitor performance in real time. The ability to spin up new instances on demand and only pay for what we use was also a big advantage.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We just did Grafana because it looked good, and the Great Expectations as our new package. We've been looking for a solution to continous monitoring and validation of data, so the Great Expectations was a good fit.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

The overall architecture of the system can be described as follows:

The starting point is the **ML developer** who works in the local development environment on their own machine. The developer uses **Docker Compose** to manage local Docker images for tasks such as pulling data, training, testing, and monitoring. Key tools include **Hydra** for configuration management, **Weights & Biases (W&B)** for experiment tracking, and **Prometheus** for monitoring.

The workflow begins with the developer making **code changes**. The process involves:

1. Pre-commit hooks for initial validation.
2. Committing the changes locally.
3. Pushing the changes to the **GitHub repository**.

Upon pushing the code, **GitHub Actions** triggers a CI/CD pipeline. This pipeline performs:

* Multi-OS testing.
* Code formatting checks.
* Linting.
* Running unit tests.

From GitHub, a **GCP trigger** initiates the build of Docker images. Two types of Docker images are created:

1. A training image for model training.
2. A FastAPI image for deployment.

The **Google Cloud Artifact Registry** stores these images and uses caching to optimize subsequent builds. The trained model is saved in the cloud, and data is mounted as needed.

The deployment process has two components:

1. **Training Deployment**: The trained model is saved and managed in the cloud.
2. **FastAPI App Deployment**: The FastAPI app is deployed for serving predictions. Internet users interact with the deployed app via SSH or direct requests.

Finally, **data versioning** and management are handled using **DVC**, with data stored in a cloud-based **Storage Bucket**. The system ensures reproducibility, scalability, and efficient model training and deployment.

Along with automation for code changes, the training and deployment processes will also run when the data is changed in DVC.

This architecture integrates multiple tools and services seamlessly, focusing on automation, monitoring, and scalability for machine learning workflows.

![Architecture Diagram](figures/architecture.png)

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

Our biggest challenge was integrating a wide range of tools—Docker, DVC, Prometheus, Grafana—while still keeping our code organized. Each service comes with its own configuration files (for instance, `docker-compose.yml`, `prometheus.yml`, and various Grafana provisioning JSONs), and we occasionally ran into version mismatches or Docker networking issues. For example, Prometheus sometimes failed to scrape the API because the container name or network alias wasn’t set up properly. We also discovered how important it is to leverage caching in Docker builds; otherwise, rebuilding images can take ages, especially when installing large packages.
We also had a day-long detour dealing with Python version differences—some team members were on Python 3.12 and others on 3.11, causing library incompatibilities. In the end, we standardized on Python 3.11.9, which resolved most of those issues.

The process of using the cloud also felt a bit slow to interact with. We had to wait for the Docker images to build, the Cloud Run service to deploy, and the Cloud Build to finish. This made it hard to iterate quickly on changes, especially when we were debugging issues with the API. We also had to be careful about costs, since we were charged for every request to the API and every minute the Cloud Run service was running. We had to be mindful of how many requests we were sending to the API and how long we were keeping the service running to avoid unexpected costs.
However this is only the feeling because it is a small scale project, in a larger project the cloud would in most cases be the faster solution.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

Student s204510 set up the initial cookiecutter structure, wrote the Dockerfiles, and integrated Great Expectations for data validation. Student s204424 handled the CI/CD pipelines on GitHub Actions, plus the DVC linking to our bucket. Student s204470 focused on the FastAPI service, Prometheus instrumentation, and Grafana dashboards. We collaborated a bit on the small amount of training code we have, just to decided which model we wanted to use. We occasionally used ChatGPT to brainstorm debugging approaches or understand error messages. We also used GitHub Copilot for small code suggestions in Python scripts. All final code was reviewed and tested by the group before merging.
