# Developer Guide

This page explains how to add new **steps** and **tests** to pelagos_py.

## What is a “step”?
A step is a stage in the pipeline that can be defined via Python, and configured via the Pipelines Config file. Examples of steps include:
- I/O Steps (e.g. reading in data using `Load OG1`->[load_data.py](https://noc-obg-autonomy.github.io/pelagos_py/api/pelagos_py/steps/custom/load_data/index.html))
- Variable Processing Steps (e.g. Adjusting existing salinity measurements using `Salinity Adjustment`->[salinity.py](https://noc-obg-autonomy.github.io/pelagos_py/api/pelagos_py/steps/custom/variables/salinity/index.html))

Steps are not limited to one per file - in fact, a single file can contain multiple steps. This also is true for configs. Any step can be called multiple
times in a single config.

As some users may want to filter out bad data before individual processing steps, this feature has been implemented through the `QCHandlerMixin`->[qc_handling.py]().
The pipeline conserves the initial dataset dimensions so the filtered data is either replaced or reinserted after the step is complete.

## What is a “test”?
A test is a sub-stage in the pipeline which strictly generates data QC. Tests are called through the `Apply QC` step [apply_qc.py]() which handles QC updating.
Examples of tests include:
- Static tests which always check the same variables (eg. `impossible date test`->[impossible_date_test.py]() which checks that the TIME variable is between 1985 and now)
- Dynamic tests which can be applied to any variable (eg. `range test`->[range_test.py]() which check that a variable exists withing a specified range)

Whilst a test can only be called once per `Apply QC`, multiple of this step can be present in a single config. 
See [all_step_config.yaml]() to see in more detail how the config for tests is defined.

## How to add a new step
A template for new data processing steps can be found in [blank_step.py]() however it is reccomended that you read the following steps to avoid implementation issues.

1. Create a new Python file in the appropriate directory under `src/pelagos_py/steps/custom/`.<br>
   **NOTE**: if you are creating a step for specific vairables then it should go in the `variables` subdirectory.
2. Define a new class for your step, inheriting from `BaseStep` and adding the @register_step decorator. This ensures that the step is discoverable by the Pipeline Manager, 
   as well as allowing you do define other classes in the same file without registering them.
   ```python
   from pelagos_py.steps.base_step import BaseStep, register_step
   
   @register_step
   class MyNewStep(BaseStep):
       ...
   ```
3. Define the step_name attribute, which is the name that will be used in the Pipelines Config file to refer to this step.
   ```python
   from pelagos_py.steps.base_step import BaseStep, register_step
   
   @register_step
   class MyNewStep(BaseStep):
   
       step_name = "My New Step"
   ```
4. Implement the `run` method, which contains the logic for your step. This method should take no arguments other than `self`, and should return a `self.context` object.
   ```python
   from pelagos_py.steps.base_step import BaseStep, register_step
   
   @register_step
   class MyNewStep(BaseStep):
       step_name = "My New Step"
        
       def run(self):
           # Your processing logic here
           return self.context        
    ```
5. Optionally, implement the `generate_diagnostics` method if your step produces any diagnostic plots or outputs.
   ```python
   from pelagos_py.steps.base_step import BaseStep, register_step
   
   @register_step
   class MyNewStep(BaseStep):
       step_name = "My New Step"
        
       def run(self):
           # Your processing logic here
           return self.context
        
       def generate_diagnostics(self):
           # Your diagnostics logic here
           pass
    ```
    There are already default methods for generating common diagnostics, such as time series plots and scatter plots. See the [diagnostics documentation](https://noc-obg-autonomy.github.io/pelagos_py/api/pelagos_py/utils/diagnostics/index.html) for more information.

6. Add the step to your Pipelines Config file, using the `step_name` you defined in step 3.
   ```yaml
    # Pipeline Configuration <- This section is only needed once at the top of the yaml file
    pipeline:
    name: "My Pipeline"
    description: "A pipeline for demonstration purposes"
   
    # Steps in the pipeline
    steps:
    - name: "My New Step"
      parameters:
        param1: value1
        param2: value2
   ```
7. Any parameters defined in the `parameters` section of the config file will be passed to your step as attributes. You can access them in your `run` method using `self.param1`, `self.param2`, etc. <br>
   **NOTE** This is handled automatically by the `BaseStep` class. More information can be found in the [BaseStep documentation](https://noc-obg-autonomy.github.io/pelagos_py/_modules/pelagos_py/steps.html).  

### Adding QC handling to a step
If you would like your step to have QC handling (pre-step filtering) then add the `QCHandlerMixin` from [qc_handling.py]() to your step class inheritance. Additionally you
   will have to include the `self.filter_qc()`, `self.reconstruct_data()`, `self.update_qc()` and `self.generate_qc({<QC_child>: [*<QC_parents>]})` methods as follows.
   ```python
   from pelagos_py.steps.base_step import BaseStep, register_step
   from pelagos_py.utils.qc_handling import QCHandlingMixin

   @register_step
   class MyNewStep(BaseStep):
       step_name = "My New Step"
        
       def run(self):
           # Before any processing happens:
           self.filter_qc() # This filters specified QC out of this steps instance of self.data and stores them separately
           
           # Your processing logic here. Always use self.data to access your processing inputs as this is what has been filtered
           # --------- EXAMPLE ---------
           self.data["C"] = self.data["A"] * self.data["B"]
           self.context["data"] = self.data
           # ---------------------------
           
           return self.context
           
           self.reconstruct_data() # Add the filtered-out data back in, or retain their replacements depending on user config
           self.update_qc() # Update the flags of the filtered data
           
           # If a new variable was added, we need to make sure it gets it's own QC column derived from its parents QC.
           # Use self.generate_qc() to do this. If no new variables were added then this is not necessary.
           # --------- EXAMPLE ---------
           self.generate_qc({"C_QC": ["A_QC", "B_QC"]})
           # ---------------------------
       
       def generate_diagnostics(self):
           # Your diagnostics logic here
           pass
   ```
To utilize QC filtering, the step config must specify `qc_handling_settings`.
   ```yaml
   # Steps in the pipeline
    steps:
    - name: "My New Step"
      parameters:
        param1: value1
        param2: value2
        # [qc_handling_settings]:
        #   Can be specified in any step that has the QC filtering functionality
        qc_handling_settings:
          # [flag_filter_settings]:
          #   {variable: flags to filter} pairs. Data that is flagged with any of the specified flags is replaced
          #   with a nan internally. All steps should be designed to operate with nans.
          flag_filter_settings:
            PRES: [3, 4]
          # [reconstruction_behaviour]:
          #   Specifies how the data will be reconstructed after processing has occured. There are two options (defaults to reinsert):
          #   "replace":  Indices where data was filtered retain their post-processing value and the original "bad data" is deleted.
          #   "reinsert": The filtered "bad data" is reinserted back into the post-processed data.
          reconstruction_behaviour: "replace"
          # [flag_mapping]:
          #   Tells the QC handler how flags should change for "bad data" indices if the pre- & postprocessing data are different.
          #   Eg. Interpolation would replace bad and missing values (3, 4, 9) with interpolated values (8).
          flag_mapping:
            3: 8
            4: 8
            9: 8
   ```

## How to add a new QC test

QC test exclusively operate on the QC flags of the data variables. This can be useful for researches post-pipeline when they want to remove bad/suspicious data or 
would like to exclude bad data from specific processing steps (see "Adding QC handling to a step" above). All tests are run through the `Apply QC` step which is
responsible for transfering the individual test results onto the existing QC columns. 

As mentioned above, there are two types of test: static & dynamic.
- static tests always check the same variable(s) and output the same variable QC results.
- Dynamic tests let the user specify which variables the test can be applied to - meaning that the QC output is not pre-determined.

A standard structure for dynamic tests is yet to be set so this section will only cover the implementation of static tests. If the dev. is interested however, examples of 
dynamics tests can be found in [range_test.py]() and [stuck_value_test.py](). 

An example template for a static test can be found in [blank_test.py](), however again it is recommended that you read the instructions below as well.

1. create your test file in the `src/pelagos_py/steps/custom/qc`.
2. Import the necessary parent classes and define your test class. Make sure it inherits from `BaseTest` and have the `@register_qc` decorator. This will allow the
    pipeline to find and register the test.
    ```python
    from pelagos_py.steps.base_test import BaseTest, register_qc, flag_cols

    @register_qc
    class MyNewTest(BaseTest):
        ...
    ```
3. Specify the following attributes:
    ```python
    from pelagos_py.steps.base_test import BaseTest, register_qc, flag_cols

    @register_qc
    class MyNewTest(BaseTest):
        test_name = "my new test"  # This is how you should call the test in config. See below...
        expected_parameters = {'A_cutoff': 1}  # These are the test parameters that we may expect from the user. The value for each key is the default.
        required_variables = ['A']  # These are the variables that are required for test execution. This is cross-referenced against the data vars in context
        qc_outputs = ['A_QC'] # These are the QC outputs. These are references that help "Apply QC" update existing QC in the data
    ```
4. Add the `return_qc()` method which is where you will implement your test algorithm. Optionally add the `plot_diagnostics()` method if you would like the test 
    to generate plots when diagnostics is true in the config.
    ```python
    from pelagos_py.steps.base_test import BaseTest, register_qc, flag_cols

    @register_qc
    class MyNewTest(BaseTest):
        test_name = "my new test"
        expected_parameters = {'A_cutoff': 1}
        required_variables = ['A']
        qc_outputs = ['A_QC']

        def return_qc(self):
            # IMPORTANT: Make sure you access the data with self.data
            # self.flags should be an xarray Dataset with data_vars that hold the "{variable}_QC" columns produced by the test
            return self.flags

        def plot_diagnostics(self):
            # Add your diagnostic plotting here
            ...
    ```
    Please note the comments in return_qc() - the data for QC should be accessed using self.data which is a xarray Dataset object. The method should also return an xarray
    Dataset (self.flags) which can contain any number of data variables, but those with the `_QC` suffix must be specified in the `qc_outputs` attribute. 
5. Finally, we just have to add our new test to the config.
    ```yaml
    - name: "Apply QC"
        parameters:
        # qc_settings can have multiple tests listed in it
        qc_settings:
            # The test name
            my new test:
            # Specify the A_cutoff setting
            A_cutoff: 100
        # If you want plotting:
        diagnostics: true
    ```