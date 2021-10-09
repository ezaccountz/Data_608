library(ggplot2)
library(dplyr)
library(plotly)
library(shiny)

df <- read.csv('https://raw.githubusercontent.com/ezaccountz/Data_608/main/module3/data/cleaned-cdc-mortality-1999-2010-2.csv')
#df <- read.csv('E:/SPS/DATA 608/Data_608/module3/data/cleaned-cdc-mortality-1999-2010-2.csv')

states <- unique(df$State)

df2 <- df %>% 
  group_by(Year,ICD.Chapter) %>% 
  summarise(State = "Average",
            Deaths = mean(Deaths),
            Population = mean(Population),
            Crude.Rate = 100000 * sum(Deaths)/sum(Population)) %>% 
  bind_rows(df)

ui <- fluidPage(
  tabsetPanel( 
    tabPanel(title = "Question 1",
      sidebarPanel(
        htmlOutput('message_q1'),
        sliderInput('year_q1','Year',min(df$Year),max(df$Year),2010,1,width=600,
                  sep = ""),
        selectInput('causes_q1', 'Causes', 
                  unique(df$ICD.Chapter), selected='Neoplasms',width = 600)
      ),
      mainPanel(plotlyOutput('plot1_q1'))
    ),
    #---------------------------------------------------------------------------
    tabPanel(title = "Question 2",
      sidebarPanel(
        htmlOutput('message_q2'),
        selectInput('causes_q2', 'Causes', 
                  unique(df$ICD.Chapter), selected='Neoplasms',width = 600),
        checkboxGroupInput("States_q2", "States",states,inline = TRUE)
      ),
      mainPanel(plotlyOutput('plot1_q2'))
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
      paste("<h4><center>No Data for<b>",input$causes_q1,
            "</b>in",input$year_q1,"</center></h4>", sep =" ") 
    }
    else {
      paste("<h4><center>The Mortality Rates for<b>",input$causes_q1,
            "</b>in",input$year_q1,"<br>(scaled by multiplying 100000)</center></h4>", sep =" ")
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
    df3 <- df2 %>%
      filter(ICD.Chapter == input$causes_q2) 
    df3
  })
  
  state2_q2 <-reactive(
    input$States_q2
  )
  
  output$message_q2 <- renderText({
    paste("<h4><center>The Mortality Rates for<b>",input$causes_q2,
                   "</b><br>(scaled by multiplying 100000) </center></h4>", sep =" ")
    
    # df2 <- data_q2()
    # if (nrow(df2) == 0) {
    #   paste("<h3><center>No Data for<b>",input$causes_q2,
    #         "</b>in",input$year_q2,"</center></h3>", sep =" ") 
    # }
    # else {
    #   paste("<h3><center>The Mortality Rates for<b>",input$causes_q2,
    #         "</b>in",input$year_q2,"</center></h3>", sep =" ")
    # }
  })
  
  output$plot1_q2 <- renderPlotly({
    
    df3 <- data_q2()
    if (nrow(df3) == 0) {
      plotly_empty(type = "scatter",width = 1,height = 1,)
    }
    else {
      temp <- filter(df3, State == "Average")
      
      fig <- plot_ly(
        x = temp$Year,
        y = temp$Crude.Rate,
        mode = 'lines+markers',
        name = "Average",
        type = "scatter"
      )
      selected_states <- state2_q2()
      for (state in states)
      {
        alpha <- ifelse(state %in% selected_states,1,0.05)
        temp <- filter(df3, State == state)
        fig <- fig %>% add_trace(x = temp$Year, y = temp$Crude.Rate, name = state, mode = 'lines+markers',opacity = alpha) 
      }
      fig
    }
  })
}

shinyApp(ui = ui, server = server)