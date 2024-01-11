def print_indices(html_text):
    tables_ = []
    s_pos = -1
    e_pos = -1
    start_tag = "<TABLE CELLSPACING="
    end_tag = "</TABLE>"
    table_text = ""
    s_pos = html_text.find(start_tag)
    e_pos = html_text[s_pos:].find(end_tag)

    while (s_pos != -1):
        print(f'{s_pos}:{e_pos}')
        table_text = html_text[s_pos:e_pos]
        tables_.append(f'{table_text}{end_tag}')

        table_text = ""
        html_text = html_text[e_pos + 8:]
        s_pos = html_text.find(start_tag)
        e_pos = html_text[s_pos:].find(end_tag)


text="""<TABLE CELLSPACING="0" CELLPADDING="0" WIDTH="100%" BORDER="0" STYLE="BORDER-COLLAPSE:COLLAPSE" ALIGN="center">
<TR>
<TD WIDTH="9%"></TD>
<TD VALIGN="bottom" WIDTH="2%"></TD>
<TD WIDTH="84%"></TD>
<TD VALIGN="bottom" WIDTH="2%"></TD>
<TD></TD>
<TD></TD>
<TD></TD></TR>
<TR>
<TD COLSPAN="3" VALIGN="bottom"><FONT SIZE="1">&nbsp;</FONT></TD>
<TD VALIGN="bottom"><FONT SIZE="1">&nbsp;&nbsp;</FONT></TD>
<TD VALIGN="bottom" COLSPAN="2" ALIGN="center" STYLE="border-bottom:1px solid #000000"><FONT STYLE="font-family:Times New Roman" SIZE="1"><B>Page</B></FONT></TD>
<TD VALIGN="bottom"><FONT SIZE="1">&nbsp;</FONT></TD></TR></TABLE>

<TABLE CELLSPACING="NEW TABLE" CELLPADDING="0" WIDTH="100%" BORDER="0" STYLE="BORDER-COLLAPSE:COLLAPSE" ALIGN="center">


<TR>
<TD WIDTH="9%"></TD>
<TD VALIGN="bottom" WIDTH="2%"></TD>
<TD WIDTH="84%"></TD>
<ONT STYLE="font-family:Times New Roman" SIZE="1"><B>Page</B></FONT></TD>
<TD VALIGN="bottom"><FONT SIZE="1">&nbsp;</FONT></TD></TR></TABLE>
"""

print_indices(text)