---
title: "Plugins in Fiori Launchpad"
description: "How to create Fiori Launchpad Plugins to enhance the Launchpad or to access the events of Launchpad."
categories: [sapui5]
tags: [fiori, pdf viewer]
---

<h3>Launchpad Plugins are used to enhance the Fiori Launchpad to extend the user experience.</h3>

<p> Before begining to demonstrate how to create Fiori Launchpad Plugin, it is better to understand what is rendering. If you have already worked on the Custom Controls then you must already be aware of this concept. Rendering, in general, refers to getting or fetching of the data. In UI5 or Javascript, rendering refers to fetching or getting the DOM elements, which Javascript then uses for Manipulation. </p>

<br><br>

<p> Now lets see, how can we create a Fiori Plugin and how does it enhance the user experience. </p>
<br><br>
<h3> Step1: Check if Extensibility extension is activated. </h3>
Extension - "SAP Fiori Launchpad Extensibility" needs to be enabled in order to create a launchpad plugin.
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/1.jpg">
<br><br>
<h3> Step2: Start Creating a new Project. Choose New->Project from Template.</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/2.jpg">

<br><br>
<h3> Step3: Give a project name.</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/3.jpg">

<br><br>
<h3> Step4: Give a plugin name and Title.</h3>
Select any of the check to get the sample code of the Plugin that needs to installed. In this case, I am going with Buttons in Me Area.
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/4.jpg">

<br> <br>

After creation, your project structure should look like this.
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/5.jpg">


<b> All the Source Code is automatically generated</b>

```javascript
init: function () {
			var rendererPromise = this._getRenderer();			
			rendererPromise.then(function (oRenderer) {
				oRenderer.addActionButton("sap.m.Button", {
					id: "myHomeButton",
					icon: "sap-icon://sys-help-2",
					text: "Help for FLP page",
					press: function () {
						MessageToast.show("You pressed the button that opens a help page.");
					}
				}, true, false, [sap.ushell.renderers.fiori2.RendererExtensions.LaunchpadState.Home, sap.ushell.renderers.fiori2.RendererExtensions.LaunchpadState.App]);
								
			});

            /* Documentation for function addAction Button or for every renderer plugin add
                addActionButton( "type_of_button", { button properties} , visible? , currentstate?, [States])

                visible --> if control is visible
                currentstate --> if true, will only be visible in Launcpad and if false, then states are considered
                states --> "Home" (Launchpad Home Page), "App" (Application Home Page)
            */
		},

_getRenderer: function () {
			var that = this,
				oDeferred = new jQuery.Deferred(),
				oRenderer;

			that._oShellContainer = jQuery.sap.getObject("sap.ushell.Container");
			if (!that._oShellContainer) {
				oDeferred.reject(
					"Illegal state: shell container not available; this component must be executed in a unified shell runtime context.");
			} else {
				oRenderer = that._oShellContainer.getRenderer();
				if (oRenderer) {
					oDeferred.resolve(oRenderer);
				} else {
					// renderer not initialized yet, listen to rendererCreated event
					that._onRendererCreated = function (oEvent) {
						oRenderer = oEvent.getParameter("renderer");
						if (oRenderer) {
							oDeferred.resolve(oRenderer);
						} else {
							oDeferred.reject("Illegal state: shell renderer not available after recieving 'rendererLoaded' event.");
						}
					};
					that._oShellContainer.attachRendererCreatedEvent(that._onRendererCreated);
				}
			}
			return oDeferred.promise();
		}

```
<br><br>
<h3> Step5: Run the project.</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/6.jpg">
<br><br>
<h3> Step6: Select Run as SAP Fiori Launchpad Sandbox</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/7.jpg">
<br><br>
<h3> Step7: In general tab, select file name "fioriSandboxConfig.json".</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/8.jpg">
<br><br>
<h3> Step8: In URL Components, enter URL hash fragment.</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/9.jpg">
<br><br>

<h3> Step9: Save and Run the application.</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/10.jpg">

<br><br>
<h3> Now, lets talk about the deployment.</h3>
<p>Once the application is deployed, plugin needs to be activated.</p>
<p> To activate, you need to register this app and provide the semantic object.</p>
<br><br>
<h3> Step10: Deploy the application of SAP Cloud Platform.</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/11.jpg">
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/12.jpg">
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/13.jpg">
<br><br>
<h3 style="color: red;"> Do not Register on Fiori Launchpad Yet as this is not an application.</h3>
<br><br>
<h3> Step11: Logon to the Cloud Platform and Enter to the Site Admin via Portal Service.</h3>
<h3> Navigate to Content Management-> Apps. And Add the application.</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/14.jpg">
<br><br>
<h3> Step12: Select the Deployed app in App Resource and App Type as Shell Plugin.</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/15.jpg">
<br><br>
<h3> Step13: Assign the catalog to this app.</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/16.jpg">
<br><br>
<h3> Step14: Now run the Website.</h3>
<img src="{{site.url}}{{site.baseurl}}/images/Fiori/20200107/17.jpg">

<br><br>
<b><i>*Phew*</i></b> This has been a long article but a good learning, right?

Happy Learning!


