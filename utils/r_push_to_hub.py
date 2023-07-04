import argparse

def parse_hf_hub():
    parser = argparse.ArgumentParser()

    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)    

    args, _ = parser.parse_known_args()

    # make sure we have required parameters to push
    if args.push_to_hub:
        if args.hub_strategy is None:
            raise ValueError("--hub_strategy is required when pushing to Hub")
        if args.hub_token is None:
            raise ValueError("--hub_token is required when pushing to Hub")

    # sets hub id if not provided
    if args.hub_model_id is None:
        args.hub_model_id = args.model_id.replace("/", "--")  

    return args

def push_to_hub(hub_args, trainer={}, model={}):
    if not hub_args.push_to_hub:
      return
    
    # save best model, metrics and create model card
    if trainer:
      trainer.create_model_card(model_name=hub_args.hub_model_id)
      trainer.push_to_hub()
    elif model:
      model.push_to_hub(hub_args.hub_model_id)
    #   model.push_to_hub("xxx/xxxxx", use_auth_token=True)