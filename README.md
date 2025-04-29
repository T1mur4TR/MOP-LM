## MOP-LM: MOrphemic Parser Language Model

### Repository Contents
 - All used datasets are either contained in the `./data` directory or are imported via the `datasets` module.
 - `common.py` contains all the used building blocks.
 - `tokenizer.py` contains the implementation of the morphemic tokenizer class.
 - `word_autoencoder.py` contains the `WordEncoder` and `WordDecoder` classes used for embedding tokenized words.
 - `dependency_parser.py` contains the `DependencyParser` class used for constructing the dependency parse tree.
 - `moplm.py` contains a simple `Transformer` class and the `MOPLM` class, which is a container for `WordEncoder`, `WordDecoder` and `Transformer` with the inference logic.
 - `training.ipynb` contains the code for training the dependency parser (Phase 1) and MOP-LM (Phase 2) with the necessary masking logic.

### Reproducing Results
Running `training.ipynb` top-to-bottom will train the dependency parser and the MOP-LM, utilising the trained parser, training doesn't stop automatically and should be stopped when the learning rate reaches a sufficiently low value.
