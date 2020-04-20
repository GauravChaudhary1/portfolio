---
title: "Handling CL_GUI_ALV_GRID events in SALV"
description: "Here is how to extend the capabilities of CL_SALV_TABLE to its full."
categories: [abap]
tags: [ooalv, salv]
---

# Intro

Have you ever tried to use an event for control CL_SALV_TABLE but unable to do so and then you had to switch to CL_GUI_ALV_GRID so that you can use wide range of events supported by SAP?

## Intuition

I believe, some developers prefer CL_SALV_TABLE as it is much easier to display the data in ALV. Because of no hassle to create a screen or create a field catalog. However, events are one area in the CL_SALV_TABLE which is lacking and making us to switch to CL_GUI_ALV_GRID. Wouldn't it be great if we could use the same events raised by CL_GUI_ALV_GRID? Well, say no more. 

## Friendly Interface - IF_ALV_RM_GRID_FRIEND

Enters friend interface of CL_GUI_ALV_GRID. All you have to do is to use a little trick, to make use of friend interface **IF_ALV_RM_GRID_FRIEND**. How? You can refer the source code below.

## Source Code

```abap
CLASS lcl_main DEFINITION
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    INTERFACES if_alv_rm_grid_friend . "This is now a mutual friend

    DATA: spfli TYPE STANDARD TABLE OF spfli,
          salv  TYPE REF TO cl_salv_table.

    METHODS: create_salv.
    METHODS: 
      double_click FOR EVENT double_click OF cl_gui_alv_grid IMPORTING e_row
                                                                       e_column
                                                                       es_row_no.
ENDCLASS.
```

```abap
CLASS lcl_main IMPLEMENTATION.
  METHOD create_salv.
    SELECT * UP TO 100 ROWS INTO CORRESPONDING FIELDS OF TABLE @spfli
    FROM spfli.

    cl_salv_table=>factory(
      IMPORTING
        r_salv_table   = salv
      CHANGING
        t_table        = spfli
    ).

    SET HANDLER double_click FOR ALL INSTANCES.


    DATA(selections) = salv->get_selections( ).
    selections->set_selection_mode(   if_salv_c_selection_mode=>cell  ). "Single row selection


    salv->display( ).

  ENDMETHOD.
  METHOD double_click.
    BREAK-POINT.
  ENDMETHOD.

ENDCLASS.

START-OF-SELECTION.

  DATA(output) = NEW lcl_main( ).
  output->create_salv( ).
```

## Conclusion
Friend in need is a friend indeed. There are a lot of events which can be used. Give it a try.
<br>Happy learning!

