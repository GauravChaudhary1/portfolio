---
title: "internationalization in Plugins for Fiori Launchpad"
description: "How to achieve internationalization in Plugin for Fiori Launcpad and how to manipulate plugin with respect to Apps in Launchpad"
categories: [sapui5]
tags: [fiori, Fiori Launchpad, Plugin]
---

<h3> Fiori Launcpad is a generic platform which is shared by multiple role experts/users. To change something in the header/footer will affect the whole launcpad and thus impacting all of the users. What if certain group of users wants that information to be shown and some don't? 
</h3>
<p>Well, yes, this can be achieved by the Role with which users are assigned. But what if Plugin needs to be deployed just for specific set of applications.</p>

<br><br>
In this article, we will be seeing below mentioned points related to Plugin Development.
<ul>
<li>Internationalization for Plugin Component.</li>

<li>Access Additional Custom Style.</li>

<li>Enable plugin only for specific set of applications.</li>
</ul>

<br><br>

<h3> Let's get started! </h3>
<h3><u>Internationalization for Plugin Component.</u></h3>
<p> We have created the Plugin Project and successfully deployed the application of ABAP or Cloud Platform. In addition to intial project structure, lets create few files <ul>
<li><b>dialog.fragment.xml</b> under folder <b>'fragment'</b> </li>
<li><b>data.json</b> under folder <b>'model'</b> </li>
<li><b>style.css</b> under folder <b>'css'</b> </li>
</ul>
<br>
Now, your project structure should look like this.<br>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200110/5.jpg">
</p>

<br><br>

In the fragment, following code is written:
<br><br>

```xml
<core:FragmentDefinition
	xmlns="sap.m"
	xmlns:core="sap.ui.core">
	<SelectDialog
		id = "idDialog"
		title="{i18n>title}"
		confirm="handleClose"
		cancel="handleClose"
		
		items="{
			path : '/data'
		}"
		>
		<StandardListItem
			title="{text}"
			description="{name}"
			icon="sap-icon://employee"
			iconDensityAware="false"
			iconInset="false"
			type="Active" />
	</SelectDialog>
</core:FragmentDefinition>
```
<br><br>
In the model/data, below data is added:
<br><br>

```json
{
"data": [
{"name": "Gaurav Chaudhary" , "text":"This is called from Data Model"}
]
}
```

<br><br>

<u><i>Important point to note here, that Fiori Launchpad has its own i18n model, and setting a model directly to Fiori Launchpad could lead to inconsistency. It is a best practice to set the i18n model directly to UI element itself.
</u></i>
<b>My i18n.properties file contains:</b><br>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200110/6.jpg">
<br><br>

```javascript
//initialize the data model

var sUrl = jQuery.sap.getModulePath("com.sap.plugin.plg1.model", "/data.json");
var oModel = new sap.ui.model.json.JSONModel(sUrl);

// Setting the i18n Model
var i18nModel = new sap.ui.model.resource.ResourceModel({
				bundleName: "com.sap.plugin.plg1.i18n.i18n"
			});

var dialogFragment = sap.ui.xmlfragment("com.sap.plugin.plg1.fragment.dialog", this);
dialogFragment.setModel(oModel);
dialogFragment.setModel(i18nModel, "i18n");			
```
<br><br>

<h3><u>Access Additional Custom Style.</u></h3>

Additional CSS can be embedded in the metadata of the Component.js using "includes".

```javascript
metadata: {
			"manifest": "json",
			includes: [
				"css/style.css" //additional CSS to design the Plugin Components.
			]
		}
```

<br><br>

<h3><u>Enable plugin only for specific set of applications.</u></h3>

Now this concept can be implemented if one can listen to events which are published by the Shell Application. Fiori Launchpad publish certain events, however, they are not documented. After thorough debugging, I was able to figure out the event which was getting published when a specific app is opened from Launchpad.
<br> <b> How the event gets published?</b>
<br>
<section>
<div>
sap.ui.core.EventBus().publish(sChannelId?, sEventId, oData?) : void
</div>
<Br>
<div> 
<ul> 
<li>sChannelId : The channel of the event to fire. If not given, the default channel is used.  </li>
<li>sEventId : The identifier of the event to fire.</li>
<li>oData: 	The parameters which should be carried by the event. </li>
</ul>
</div>
</section>

<br> <b> How the event gets subscribed?</b>
<br>
<section>
<div>
sap.ui.core.EventBus().subscribe(sChannelId?, sEventId, fnFunction, oListener?) : sap.ui.core.EventBus
</div>
<Br>
<div> 
<ul> 
<li>sChannelId : The channel of the event to subscribe to. If not given, the default channel is used.  </li>
<li>sEventId : The identifier of the event to listen for.</li>
<li>fnFunction: The handler function to call when the event occurs. This function will be called in the context of the oListener instance (if present) or on the event bus instance.</li>
<li>oListener: The object that wants to be notified when the event occurs (this context within the handler function).</li>
</ul>
</div>
</section>

<br> Now let's see that event.

<br><br>

```javascript
sap.ui.getCore().getEventBus().subscribe(
				"launchpad",
				"appOpened",
				function_to_listen_event,
				this
			);
```

<br><br>
<h3> Now let's see the code in the component.js! I have enabled a custom button only for app 'RTA Demo App'.</h3>


<br><br>

```javascript
metadata: {
			"manifest": "json",
			includes: [
				"css/style.css"
			]
		},

		init: function () {
			var rendererPromise = this._getRenderer();
			this.i18nModel = new sap.ui.model.resource.ResourceModel({
				bundleName: "com.sap.plugin.plg1.i18n.i18n"
			});

			this.setModel(this.i18nModel, "i18n");
			var sUrl = jQuery.sap.getModulePath("com.sap.plugin.plg1.model", "/data.json");
			this.oModel = new sap.ui.model.json.JSONModel(sUrl);
			var that = this;
			this._subscribeEvents();
			rendererPromise.then(function (oRenderer) {
				that.oRenderer = oRenderer;
				oRenderer.addActionButton("sap.m.Button", {
					id: "myHomeButton",
					icon: "sap-icon://sys-help-2",
					text: "Custom Plugin Button",
					press: function () {
						if (!this.dialogFragment) {
							this.dialogFragment = sap.ui.xmlfragment("com.sap.plugin.plg1.fragment.dialog", this);
							this.dialogFragment.setModel(that.oModel);
							this.dialogFragment.setModel(that.i18nModel, "i18n");
						}

						this.dialogFragment.open();
					}
				}, true, false, [sap.ushell.renderers.fiori2.RendererExtensions.LaunchpadState.App]);

			});
		}
```


```javascript
_subscribeEvents: function(  ){
		   sap.ui.getCore().getEventBus().subscribe(
				"launchpad",
				"appOpened",
				this.onAppOpened,
				this
			);
			sap.ui.getCore().getEventBus().subscribe(
				"sap.ushell",
				"appOpened",
				this.onAppOpened,
				this
			);
		}
```

```javascript
onAppOpened: function( e1, e2, appMeta, e4 ){
			if(appMeta["text"] === "RTA Demo App"){
				this.oRenderer.showActionButton(["myHomeButton"], false, ["app"]);	
			}else
			{
				this.oRenderer.hideActionButton(["myHomeButton"], false, ["app"]);
			}
		}
```

<h3> Now let's see the results!</h3>

<b>After running the launchpad, no custom button is visible in the Me area.</b>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200110/1.jpg">

<br>

<b>After opening the app "Default Application",still no custom button is visible in the Me area.</b>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200110/2.jpg">

<br>

<b>After opening the app "RTA Demo App", custom button is visible in the Me area.</b>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200110/3.jpg">

<br>

<b>Performing the action on button, we can see our i18n translation and Data Model Binding.</b>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200110/4.jpg">

<br>

If you feel this informative, please do share it with other and let me know in the comments!
<br><i>Happy Learning.</i>

