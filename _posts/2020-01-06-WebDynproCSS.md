---
title: "Embed custom styles in WebDynpro"
description: "Adding custom css to Webdynpro to enhance the visualization of WebDynpro Applications. "
categories: [abap]
tags: [webdynpro, css]
---
<p>
There are various blogs over the internet demonstrating the usage of Custom CSS in the webdynpro applications. To find the specific use case over the internet is sometimes very hard, when the use case you are referring to is quite different.</p>

<b>Here goes my business requirement:</b>

<p>On the left side of screen, there are two UI elements (TextEdit) for input (<i>Subject & Text</i>) and on the right side, there are buttons which will populate the either of the two inputs.</p>

<br><br><br><br>
<img src="{{site.url}}{{site.baseurl}}/images/ABAP/20200106/1.jpg">
<br><br><br><br>
<p>However, in this case, there is no option to get the trigger where exactly user wants to put the text. Is it in Input1 or Input2?
Since, in Webdynpro, I could not access 'onClick' event of the UI element and neither can we engage with the DOM elements directly.</p>

<br><br><br><br>
<img src="{{site.url}}{{site.baseurl}}/images/ABAP/20200106/2.gif">
<br><br><br><br>
<p>So, I thought of Changing the Label to Button so that it triggers an event where I could store which Input is actually requested. Let's say if I click on Subject, then system should know that I want to access the Subject and all the buttons on right side should fill the subject.</p>
<br><br>
<b>Something like this:</b>
<br><br><br><br>
<img src="{{site.url}}{{site.baseurl}}/images/ABAP/20200106/4.jpg">
<br><br><br><br>
<h3>Wait! Its not resolved yet, adding a button instead of Label distorts the visual harmonization and to some organization that is more important escpecially SAP.</b>
<br><br><br><br>
<h3>CSS comes to the rescue.</h3>
<br><br><br><br>
<p>If I am able to convert a button into label with all its properties, voilà, its all sorted for me then. And great thing, I did.</p>
<br><br><br><br>
```abap
  " Component Controller -> WDDOINIT  
  DATA(lo_custom_style_manager) = wd_this->wd_get_api( )->get_application( )->get_custom_style_manager( ).

  DATA lo_btn_style_properties TYPE if_wd_custom_style=>t_style_properties.

  lo_btn_style_properties = VALUE #(
                                      ( name = 'borderColor' value = 'transparent!important' )
                                      ( name = 'hoverBorderColor' value = 'transparent!important' )
                                      ( name = 'backgroundColor' value = 'transparent!important' )
                                      ( name = 'hoverBackgroundColor' value = 'transparent!important' )
                                      ( name = 'fontColor' value = '#666!important' ) "label Color
                                    ).

  DATA(lo_btn_custom_style) = lo_custom_style_manager->create_custom_style( style_class_name = 'myCustomButton'
                                                                      library_name     = 'STANDARD'
                                                                      element_type     = 'BUTTON'
                                                                      style_properties = lo_btn_style_properties ).

  lo_custom_style_manager->add_custom_style( lo_btn_custom_style ).

```

<br><br><br><br>
<p>
Theme which is created in the WDDOINIT method, then needs to be passed to the Button's property 'styleClassName'.
</p>

<br><br>
Results can be seens here.
<br><br><br><br>
<img src="{{site.url}}{{site.baseurl}}/images/ABAP/20200106/3.jpg">
<br><br><br><br>