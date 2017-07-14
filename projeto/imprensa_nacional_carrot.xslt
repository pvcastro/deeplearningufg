<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
     xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
     xmlns:media="http://search.yahoo.com/mrss">

  <xsl:output indent="yes" omit-xml-declaration="no"
       media-type="application/xml" encoding="UTF-8" />

  <xsl:template match="/">
    <searchresult>
      <xsl:apply-templates select="/response/result/doc" />
    </searchresult>
  </xsl:template>

  <xsl:template match="doc">
    <document>
      <title><xsl:value-of select="api_Titulo_tg" /></title>
      <snippet>
        <xsl:value-of select="api_Conteudo_tg" />
      </snippet>
    </document>
  </xsl:template>
</xsl:stylesheet>
