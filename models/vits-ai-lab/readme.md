source: https://repository.clarin.lv/repository/xmlui/handle/20.500.12574/89

# Latvian Male VITS Text-to-Speech Model (vers. 2023)

Model trained using the [VITS toolkit](https://github.com/jaywalnut310/vits/tree/2e561ba58618d021b5b8323d3765880f7e0ecfdb).

Phonetic transcription generated in the IPA format using the [Latvian Phonetic Transcriber](https://github.com/Skriptotajs/PhoneticTranscriber/tree/ecf0e45b91b4ad5ba293ff7cca9073dba4cfafba).

Files included in the directory:

* `VITS_Latvian_G.pth` - the model file;
* `Latvian.json` - the configuration file;
* `symbols.py` - the modified symols file (replaces `text/symbols.py`);
* `phoneme_map.tsv` - the mapping used to convert IPA phonemes to symbols passed to VITS.
