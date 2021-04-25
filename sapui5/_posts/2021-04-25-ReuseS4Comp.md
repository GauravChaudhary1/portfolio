---
title: "Reuse S/4 Applications"
description: "Reuse S/4 HANA Applications in Custom Applications"
categories: [sapui5]
tags: [S/4HANA]

---

# Reuse S/4 HANA Attachment Framework in Custom Application

SAP S/4 HANA uses a generic attachment framework which is shared among most of the applications using attachment services, like Purchase Orders. Wouldn't it be cool if we could leverage the same in our custom applications. 

## How will it help?

 - It allows modularity and reusability throughout the application.
 - It ensures we stay updated with any standard application framework changes.
 - It provides draft compatibility services, i.e., attachment gets saved as draft first until they are saved explicitly to the database.
 - No hassle of maintaining either oData service or actual application.

##  Okay, then how do we begin?

First things first, we will understand few technical details which application has to offer.
Standard Application which we need to re-use: 
**sap.se.mi.plm.lib.attachmentservice**

This service offers three different things:

 1. Attachment Library Services as a Control
 2. Free Style UI5 Application embedding
 3. Smart Template Fiori Application embedding

Out of these, we will be focusing mainly on the Fiori Application embedding in this blog.

## Designing a custom framework for backend
Since every standard application handles storing and authorization separately however, we can design one custom framework which can be reused in every custom application.
What we need?

 1. We need a BADI for Handling Authorization(this is important)
 2. We need a BADI if we want to provide any addition for draft persistence.
 3. Likewise, we need a BADI if we want logic to persist in the data.
 4. We need a custom BO for attachment.


## Creating Custom BO

Navigate to tcode SWO1 and create the BO like below:

![](/images/Fiori/20210425/1.png)

We do not need to generate the BO as we just the reference for SAP DMS(Document Management System).
> We can generate the BO by providing necessary details for Modeled Objects. But for this blog and framework we do not need this.

## Implementing a BADI

In this blog I will implement BADI only for handling Authorization.
Enhancement Spot: **ES_CV_ODATA_ATTACHMENTS**
This offers three BADI, and we will implement BADI **BADI_CV_ODATA_ATTACHMENTS_AUTH**

This is "multiple use" BADI and therefore we need to provide a filter for our custom applications.
We will only provide the BO name which we created in last section as OBJECT_TYPE for the filter.

![](/images/Fiori/20210425/2.png)

Now we need to implement the interface methods for BADI. We are provided with two methods

 - **IF_EX_CV_ODATA_ATTACHMENT_AUTH~CHECK_AUTHORIZATION** - This is a method which checks whether user is authorized for the service or not.
 - **IF_EX_CV_ODATA_ATTACHMENT_AUTH~CHECK_USER_AUTHORIZATION** - This methods checks for a user which actions are authorized. 
Ofcourse we would need to implement the logic for both.

For the first method, if you do not wish to implement authorization logic, please make sure to clear the returning variable.

```abap
method IF_EX_CV_ODATA_ATTACHMENT_AUTH~CHECK_AUTHORIZATION.  
  " Provide authorization logic here
  " Else, just clear the returning variable
CLEAR :cv_no_authorization.  
  
endmethod.
```

## Configuring the Frontend

### Configuring the manifest.json

Add lib "sap.se.mi.plm.lib.attachmentservice" as dependency.

```json
"sap.ui5": {
		"dependencies": {
		"minUI5Version": "1.65.0",
		"libs": {
			"sap.ui.core": {
				"lazy": false
			},
			"sap.ui.generic.app": {
				"lazy": false
			},
			"sap.suite.ui.generic.template": {
				"lazy": false
			},
			"sap.se.mi.plm.lib.attachmentservice": {
			}

		"components": {}
		}
}		
```
Add below to Object Page of the main entity. This makes sure, that the component is added as a section.

```json
"embeddedComponents":{
	"simple::Attachments": {
		"id": "simple::Attachments",
		"componentName": "sap.se.mi.plm.lib.attachmentservice.attachment.components.stcomponent",
		"title": "{{Attachments}}",
		"settings": {
		"mode": "{= ${ui>/editable}?'C':'D'}",
		"objectType": "ZATTACH",
		"objectKey": "{parts:[{path:'Travreq'},{path:'DraftUUID'}],formatter:'com.sap.fiori.travel.model.formatter.returnAttachmentKey'}"
		}
	}
}
```

![](/images/Fiori/20210425/3.png)


and as for the formatter function

```javascript
jQuery.sap.declare("com.sap.fiori.travel.model.formatter");

com.sap.fiori.travel.model.formatter = {

returnAttachmentKey:  function (travreq, DraftUUID) {
	var  objectKey = "";
	if (travreq !== undefined && travreq !== null && travreq !== "00000000") {
		objectKey = travreq;
	} else  if (DraftUUID !== undefined && DraftUUID !== null && DraftUUID !== "") {
		objectKey = DraftUUID.replace(/[^a-zA-Z0-9]/g, "");
	}
	return  objectKey;
	}
}

```

## For final save to DB, use below code snipped to convert objects from Draft to Active on Save Handler

```abap
DATA objectkey TYPE objky.  
DATA temp_objectkey TYPE objky.  
DATA(attachment_api) = cl_odata_cv_attachment_api=>get_instance( ).

" ObjectKey : Key with which draft should be converted.
" Temp_objectkey : Attachment saved with Draft Id's

CALL METHOD attachment_api->if_odata_cv_attachment_api~save  
EXPORTING  
iv_objecttype = 'ZATTACH'  
iv_objectkey = objectkey  
iv_objecttype_long = 'ZATTACH'  
iv_temp_objectkey = temp_objectkey  
iv_no_commit = abap_true  
IMPORTING  
ev_success = DATA(ok)  
et_messages = DATA(messages).
```

# Now, let's see the output

Above configurations have added a reusable component as a section.

![](/images/Fiori/20210425/4.png)

You can go ahead and upload the attachments and it will get saved as the draft. Once handled, it can also be converted to Active entity.

![](/images/Fiori/20210425/5.png)


Let me know if this was helpfull to you.

[For more details check here](https://help.sap.com/viewer/36802406aebb4b96b1598246e1d316ee/2020.000/en-US/94a0ae633f6f4bfcbc7e1cb255eb908f.html)

Happy Learning!