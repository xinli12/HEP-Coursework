**Usage Note for `events.out.tfevents.*` File**

This file is a TensorFlow event log used for visualising training metrics in TensorBoard.  
To view it, run the following command in the log directory:

```bash
tensorboard --logdir=.
```

Then open your browser and go to `http://localhost:6006`.