---
title: "PDF Viewer Control working in Fiori Application"
description: "PDF Viewer Control Working in the Fiori Application."
categories: [sapui5]
tags: [fiori, pdf viewer]
---

<p>
PDFViewer Control helps in viewing a PDF document within a WebPage, without having the need to download the document first and then view it. There could be multiple sources for the PDF from where the document can be loaded. Generally, I could figure out three kinds of source origin for the document.
1. Access PDF File which has different origin. Example: Sharepoint, server.

2. Access PDF File which is on same project directory.

3. Access PDF File from the database using oData Services.

Lets see how to get these working.
 </p>



<h3> 1. Access PDF File which has different origin. </h3>
<p>
For the Files with different origin, absolute path needs to provided to the control PDF Viewer. There could be usecases where files such as Policy Document, Help Document, etc. need to shown. This seems pretty straight forward.
Below pen shows the result of using the absolute path for Control PDFViewer.
</p>

<p class="codepen" data-height="265" data-theme-id="default" data-default-tab="result" data-user="zgaur" data-slug-hash="BayWOLN" style="height: 265px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; border: 2px solid; margin: 1em 0; padding: 1em;" data-pen-title="PDF Viewer">
  <span>See the Pen <a href="https://codepen.io/zgaur/pen/BayWOLN">
  PDF Viewer</a> by Gaurav Chaudhary (<a href="https://codepen.io/zgaur">@zgaur</a>)
  on <a href="https://codepen.io">CodePen</a>.</span>
</p>
<script async src="https://static.codepen.io/assets/embed/ei.js"></script>

<h3> 2. Access PDF File which is on same project directory. </h3>
<p>
For the Files within the same project directory, relative path can be provided to the control PDF Viewer. For such cases, if App Descriptor needs to be shown, then this would be agood approach.

I believe, till now, you have create a SAPUI5 project in WEB IDE or any IDE and started working on the PDF Viewer control.
I have added a file <u>"sample.pdf"</u> in the folder "models" which is under my project and then I provided a relative path to this file to PDF Viewer control.
</p>

In the View, I have added this part and Binded to a local JSON Model.

```xml
			<PDFViewer source="{/Source}" title="SAP" height="600px">
				<layoutData>
					<FlexItemData growFactor="1" />
				</layoutData>
			</PDFViewer>

```

In the Controller,

```javascript
onInit: function () {
				this._sValidPath = "model/sample.pdf";				
				this._oModel = new JSONModel({
					Source: this._sValidPath,
					Title: "My Custom Title",
					Height: "600px"
				});
				this.getView().setModel(this._oModel);								
        },        
```

Here is the output.
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/PDFViewer/localmodel.png">

Note the name of the file, sample.pdf, in the left-top corner of the control. This is the name of the file which control picks from the file source. This could be different in other scenario's like fetching file from database unless explicitely handled.

<h3> 3. Access PDF File from the database using oData Services. </h3>
<p>
This could get little tricky, however, this is as easy as any other source, if you have your logic to fetch file is sorted.
</p>
Let's see the View:

```xml
	<PDFViewer source="/sap/opu/odata/SAP/ZTEST_SRV/FileSet(' ')/$value" title="SAP" height="600px">
				<layoutData>
					<FlexItemData growFactor="1" />
				</layoutData>
			</PDFViewer>
```

In the controller,

```javascript
onInit: function () {
                //Not even needed, as I have set my model to the Component.js
				this.oDataModel = this.getOwnerComponent().getModel();
		},

```

Now, lets see the logic used to fetch the file from database using oData.
I have added the logic in method GET_STREAM of the data provider class.

```abap
    CALL METHOD lo_document->download              
              RECEIVING
                rs_document_content = ls_document_content.

            FIELD-SYMBOLS <ls_stream> TYPE /iwbep/cl_mgw_abs_data=>ty_s_media_resource.
            CREATE DATA er_stream TYPE /iwbep/cl_mgw_abs_data=>ty_s_media_resource.
            ASSIGN er_stream->* TO <ls_stream>.

            <ls_stream>-mime_type = ls_document_content-mime_type.
            <ls_stream>-value = ls_document_content-content.

            DATA(lv_encoded_filename) = escape( val = ls_document_content-file_name format = cl_abap_format=>e_url ).
            lv_utf8_encoded_filename = lv_encoded_filename.
            REPLACE ALL OCCURRENCES OF ',' IN lv_utf8_encoded_filename WITH '%2C'. "#EC NOTEXT
            REPLACE ALL OCCURRENCES OF ';' IN lv_utf8_encoded_filename WITH '%3B'. "#EC NOTEXT
            "This is the important part here.
            ls_header-name  =  'Content-Disposition'.           "#EC NOTEXT
            ls_header-value = 'inline; filename=' && lv_encoded_filename && ';'.

            set_header( ls_header ).
```
In a regular HTTP response, the Content-Disposition response header is a header indicating if the content is expected to be displayed inline in the browser, that is, as a Web page or as part of a Web page, or as an attachment, that is downloaded and saved locally.

With the above header set, you will get the output as:
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/PDFViewer/odatamodel.png">

I believe, this might help resolve any issue that you are facing. Feel free, to share this.
<i> Happy Learning</i>
