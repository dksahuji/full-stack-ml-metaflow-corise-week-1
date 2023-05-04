"""

Template for writing Metaflows

"""

from metaflow import FlowSpec, step, current, card, Parameter
from functools import wraps

class skip:
    """
    A Flow decorator to skip a step in flow by checking on the check argument
    Args:
        check: condition to check for whether to skip the current step,
        condition have to be set to self attribute.
        next: name of the next step, after skipping
    """

    def __init__(self, check: str, next: str) -> None:
        self.check = check
        self.next = next

    def __call__(self, f):
        @wraps(f)
        def func(s):
            if getattr(s, self.check):
                s.next(getattr(s, self.next))
            else:
                return f(s)
        return func



class Template_Flow(FlowSpec):
    """
    Template for Metaflows.
    You can choose which steps suit your workflow.
    We have included the following common steps:
    - Start
    - Process data
    - Data validation
    - Model configuration
    - Model training
    - Model deployment
    """
    
    to_skip = Parameter("to_skip", default=False, type=bool)

    @card
    @step
    def start(self):
        """
        Start Step for a Flow;
        """
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)

        # Call next step in DAG with self.next(...)
        self.next(self.process_raw_data)

    @step
    def process_raw_data(self):
        """
        Read and process data
        """
        print("In this step, you'll read in and process your data")

        self.next(self.data_validation)

    @step
    def data_validation(self):
        """
        Perform data validation
        """
        print("In this step, you'll write your data validation code")

        self.next(self.get_model_config, self.get_another_model_config)

    @step
    def get_model_config(self):
        """
        Configure model + hyperparams
        """
        print("In this step, you'll configure your model + hyperparameters")
        self.next(self.train_model)
    
    @skip(check="to_skip", next="train_model")
    @step
    def get_another_model_config(self):
        """
        Configure model + hyperparams
        """
        print("skip this step. ")
        self.next(self.train_model)

    @step
    def train_model(self, inputs):
        """
        Train your model
        """
        print("In this step, you'll train your model")

        self.next(self.deploy)

    @step
    def deploy(self):
        """
        Deploy model
        """
        print("In this step, you'll deploy your model")

        self.next(self.end)
    
    @step
    def end(self):
        """
        DAG is done! Congrats!
        """
        print("DAG ended! Woohoo!")


if __name__ == "__main__":
    Template_Flow()
