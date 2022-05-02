using PrettyTables
using RegressionTables

pretty_table(P_r,backend = :latex, noheader = :true, formatters = ft_printf("%6.4f"))
pretty_table(P_r,backend = :latex, noheader = :true, formatters = ft_printf("%6.4f"))


plot(p095)
plot(p099)


regtable(tauc,rouw;renderSettings = latexOutput())
