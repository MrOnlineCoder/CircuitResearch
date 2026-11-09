
== Формули

Закон Ома та потужність записуються як
$ U = I R, quad P = U I = I^2 R. $ <ohm>

Посилання на формулу: @ohm.

== Фрагменти коду

Вбудований код у тексті: `to_spice(components)`. Блок коду з підсвіткою синтаксису
наведено в @lst-spice.

#figure(
  ```python
  def to_spice(components):
      # Серіалізація у SPICE-нетліст
      return "\n".join(f"{c.name} {c.a} {c.b} {c.value}" for c in components)
  ```,
  caption: [Серіалізація компонентів у SPICE],
) <lst-spice>

Код також можна підключати з окремого файлу (@lst-file):

#figure(
  raw(read("code/netlist.py").trim(), lang: "python", block: true),
  caption: [Вміст файлу `code/netlist.py`],
) <lst-file>

Результат — SPICE-нетліст:

```spice
R1 in out 1k
C1 out 0 10n
```

== Таблиці

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    table.header[*Метод*][*Тип*][*Точність, %*],
    [Image2Net], [конвеєр], [80,77],
    [SINA], [конвеєр], [96,67],
  ),
  caption: [Порівняння методів за даними огляду @circuitsurvey],
) <tab-methods>

= Висновки

Див. таблицю @tab-methods.
