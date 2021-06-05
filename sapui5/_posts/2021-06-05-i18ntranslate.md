---
title: "Auto-translations of UI5/Fiori Apps "
categories: [sapui5]
tags: [S/4HANA]

---


# Auto-translations of UI5/Fiori Apps  

  

##  Leverage NLP to auto translate i18n files in any desired language

As a Frontend developer, we often have to translate the i18n files to more than 1 language(sometimes). And there are a lot of properties, generally, used in 1 single file. Won't it be a good idea if we could just automate this? Though SAP provides a translation hub which can help but that still is a lot of work. Let's look at briefly what the idea is.
Also, source can be found [here](https://github.com/GauravChaudhary1/auto-translation).
  
## NLP Python Library & NodeJS 

I have used a pre-delivered library, since it provides most of the languages and has a good trained model.
[Google-trans-new](https://pypi.org/project/google-trans-new/) - This provides over 30+ language translations and works pretty well.

## How it works?

Python script checks the i18n file and access all the key fields and translates into desired language and generate a new file if it not already exists.
Take a look at the file [translate.py](https://github.com/GauravChaudhary1/auto-translation/blob/main/translate.py)

Sample working of the translator.

```python
from google_trans_new import google_translator  

translator = google_translator()  
translate_text = translator.translate('Main',lang_tgt='hi')  
print(translate_text)
-> मुख्य
```

In manifest.json, I have added a key "targetLanguages" which is then collected by translate.py 


```json

"targetLanguages":[
			"ru",
			"de",
			"fr",
			"hi",
			"zh"
			]

```

Either you can run the command to execute python file. or you can provide it in NPM command. For example, in package.json:

```json
"scripts": {
"deploy": "ui5 build preload --clean-dest --config ui5-deploy.yaml --include-task=generateManifestBundle generateCachebusterInfo && rimraf archive.zip",

"translate": "python translate.py"
```

With earlier UI5 tooling versions, encoding was mandated to ISO-8859-1 which is what SAP uses internally to encode or decode. However, with new UI5 tooling version (>1.74) this is resolved. You can specify, encoding preference. [More Details](https://sap.github.io/ui5-tooling/pages/Configuration/)

## Running command - NPM run translate
i18n.properties

```properties
title=Title
appTitle=Main
appDescription=App Description

step1=Select Action
step2=Select Action
```

Hindi Translation(i18n_hi.properties)

```properties
title=शीर्षक
appTitle=मुख्य
appDescription=ऐप विवरण

  

step1=कार्रवाई चुनें
step2=कार्रवाई चुनें
```


Russian Translation(i18n_ru.properties)

```properties
title=Заголовок
appTitle=Основной
appDescription=Описание приложения

step1=Выберите действие
step2=Выберите действие
```

Now it has become a lot easier to translate, I just need to run only 1 command and best part is code is reusable, and I can use it for not just UI5/Fiori but to any frontend apps, like Angular, React, even Python itself.


### PS: I have used VSCode for this, however, it is also possible in Business Application Studio
Checkout this [link](https://blogs.sap.com/2021/01/21/xtending-business-application-studio-4-of-3/) to get started on installing python to BAS.