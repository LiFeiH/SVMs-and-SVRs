gg = function(p){
  p + theme(axis.line = element_line(colour = "gray1",
    size = 0.2, linetype = "solid"), panel.grid.major = element_line(colour = "gray90"),
    panel.grid.minor = element_line(colour = "gray90"),
    axis.text = element_text(family = "serif",
         size = 18, face = "bold"), axis.text.x = element_text(family = "serif",
         size = 18), panel.background = element_rect(fill = "white"),
     legend.position = "none") +labs(x = NULL, y = NULL, colour = NULL)
}