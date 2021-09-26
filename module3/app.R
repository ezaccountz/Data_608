library(ggplot2)
library(dplyr)
library(plotly)
library(shiny)

df <- read.csv('https://raw.githubusercontent.com/ezaccountz/Data_608/main/module3/data/cleaned-cdc-mortality-1999-2010-2.csv')
#df <- read.csv('E:/SPS/DATA 608/Data_608/module3/data/cleaned-cdc-mortality-1999-2010-2.csv')


ui <- fluidPage(
  tabsetPanel( 
    tabPanel(title = "Question 1",
      sidebarPanel(
        htmlOutput('message_q1'),
        sliderInput('year_q1','Year',min(df$Year),max(df$Year),2010,1,width=600,
                  sep = ""),
        selectInput('causes_q1', 'Causes', 
                  unique(df$ICD.Chapter), selected='Neoplasms',width = 600)),
        mainPanel(plotlyOutput('plot1_q1')
      )
    ),
    #---------------------------------------------------------------------------
    tabPanel(title = "Question 2",
      sidebarPanel(
        htmlOutput('message_q2'),
        sliderInput('year_q2','Year',min(df$Year),max(df$Year),2010,1,width=600,
                  sep = "",animate=TRUE),
        selectInput('causes_q2', 'Causes', 
                  unique(df$ICD.Chapter), selected='Neoplasms',width = 600)),
        mainPanel(plotlyOutput('plot1_q2')
      )
    )
  )
)


server <- function(input, output, session) {

  data_q1 <- reactive({
    df2 <- df %>%
       filter(Year == input$year_q1) %>%
       filter(ICD.Chapter == input$causes_q1) %>%
       arrange(Crude.Rate) 
    df2$State <- factor(df2$State, levels = unique(df2$State))
    df2
  })
  
  output$message_q1 <- renderText({
    
    df2 <- data_q1()
    if (nrow(df2) == 0) {
      paste("<h3><center>No Data for<b>",input$causes_q1,
            "</b>in",input$year_q1,"</center></h3>", sep =" ") 
    }
    else {
      paste("<h3><center>The Mortality Rates for<b>",input$causes_q1,
            "</b>in",input$year_q1,"</center></h3>", sep =" ")
    }
  })

  output$plot1_q1 <- renderPlotly({

    df2 <- data_q1()
    if (nrow(df2) == 0) {
      plotly_empty(type = "bar",width = 1,height = 1,)
    }
    else {
    plot_ly(x = df2$Crude.Rate,
            y = df2$State,
            orientation='h',
            width = 900,
            height = 750,
            type = "bar") %>% 
        layout(
          yaxis = list(tickfont = list(size = 11)))
    }
  })
  
#-------------------------------------------------------------------------------
  
  data_q2 <- reactive({
    df2 <- df %>%
      filter(Year == input$year_q2) %>%
      filter(ICD.Chapter == input$causes_q2) 
    df2
  })
  
  max_rate_q2 <-reactive(
    df %>%
      filter(ICD.Chapter == input$causes_q2) %>% 
      select(Crude.Rate) %>% 
      max()
  )
  
  output$message_q2 <- renderText({
    
    df2 <- data_q2()
    if (nrow(df2) == 0) {
      paste("<h3><center>No Data for<b>",input$causes_q2,
            "</b>in",input$year_q2,"</center></h3>", sep =" ") 
    }
    else {
      paste("<h3><center>The Mortality Rates for<b>",input$causes_q2,
            "</b>in",input$year_q2,"</center></h3>", sep =" ")
    }
  })
  
  output$plot1_q2 <- renderPlotly({
    
    df2 <- data_q2()
    if (nrow(df2) == 0) {
      plotly_empty(type = "bar",width = 1,height = 1,)
    }
    else {
      mean_mortality <- 100000*sum(df2$Deaths)/sum(df2$Population)
      df2$colors <- ifelse(df2$Crude.Rate > mean_mortality, "#349ceb", "#34dbeb")
      
      a <- list(
        x = mean_mortality,
        text = "average",
        xref = "x",
        yref = "paper",
        showarrow = TRUE,
        arrowhead = 1,
        ax = 50,
        ay = -50,
        font = list(size = 20,bold = TRUE)
      )
      
      vline <- list(
        type = "line", 
        y0 = 0, 
        y1 = 1, 
        yref = "paper",
        x0 = mean_mortality, 
        x1 = mean_mortality, 
        line = list(color = "black")
      )
      
      plot_ly(
        x = df2$Crude.Rate,
        y = df2$State,
        marker = list(color = df2$colors),
        orientation='h',
        width = 900,
        height = 750,
        type = "bar"
      ) %>% 
        layout(shapes = vline,
               xaxis = list(range=c(0,max_rate_q2())),
               yaxis = list(tickfont = list(size = 11)),
               annotations = a
        )
    }
  })
}

shinyApp(ui = ui, server = server)