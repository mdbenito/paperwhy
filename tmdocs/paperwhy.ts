<TeXmacs|1.99.4>

<style|source>

<\body>
  <\active*>
    <\src-title>
      <src-style-file|paperwhy|1.0>

      <src-purpose|Style for paperwhy posts>
    </src-title>
  </active*>

  <use-package|generic>

  <\active*>
    <\src-comment>
      Style parameters.
    </src-comment>
  </active*>

  <assign|font|stix>

  <assign|font-base-size|11>

  <assign|math-font|math-stix>

  <\active*>
    <\src-comment>
      Macro definitions.
    </src-comment>
  </active*>

  \;

  <assign|quotation|<\macro|body>
    <\padded>
      <\indent-both|<value|quote-left-indentation>|<value|quote-right-indentation>>
        <surround|<yes-indent>||<em|<arg|body>>>
      </indent-both>
    </padded>
  </macro>>
</body>

<initial|<\collection>
</collection>>