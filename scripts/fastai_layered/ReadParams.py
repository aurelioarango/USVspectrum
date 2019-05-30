#Aurelio Arango
#Date 11/28/2018


import sys, getopt


def train_read_params(argv):

    load_path = 'F:\\USV_DATA\\data_512\\small'
    save_path = 'F:\\USV_DATA\\data_512\\models'
    model='resnet18'
    model_name = 'default'
    epochs = 10
    print (argv)

    try:
        opts, args = getopt.getopt(argv, "hH:l:s:m:e:n:",["help","loadpath","savepath", "model", "epochs","name"])
    except getopt.GetoptError:
        print("main_train.py -m <model>")
        sys.exit(2)
    print("opts: ",opts)

    #if len(opts) == 0:
    #    sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h","-H", "help"):
            print("-m model to use <resnet18, resnet34, resnet50, resnet101, resnet152>")
            print("-l <load path>")
            print("-s <save path>")
            print("-e <number epochs>")
            print("-n <model name>")
            sys.exit(2)
        elif opt in ("-l","--loadpath"):
            load_path = arg
        elif opt in ("-s","--savepath"):
            save_path = arg
        elif opt in ("-m","--model"):
            model = arg
        elif opt in ("-e","--epochs"):
            print("arg", arg)
            epochs = int(arg)
        elif opt in ("-n","--name"):
            model_name = arg
        else:
            print("-m model to use <resnet18, resnet34, resnet50, resnet101, resnet152>")
            print("-l <load path>")
            print("-s <save path>")
            print("-e <number epochs>") 
            sys.exit(2)

    print("Load Path: ", load_path)
    print("Save Path: ", save_path)
    print("Model: ", model)
    print("Epochs: ", epochs)
    print("Model Name", model_name)

    return model, load_path, save_path, epochs, model_name


#def read_test_params(argv):
"""Function to read parameters for testing"""


