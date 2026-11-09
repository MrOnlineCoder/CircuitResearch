// Шаблон оформлення роботи.
// Використовує лише вбудовані можливості Typst — без зовнішніх пакетів.

#let report(
  title: "",
  subtitle: none,
  author: "",
  supervisor: none,
  university: none,
  city: "Київ",
  year: datetime.today().year(),
  body,
) = {
  set document(title: title, author: author)

  // ---------- Сторінка ----------
  set page(
    paper: "a4",
    margin: (left: 30mm, right: 15mm, top: 20mm, bottom: 20mm),
    numbering: "1",
    number-align: center,
  )

  // ---------- Текст (українська мова) ----------
  // Перший доступний шрифт буде використаний; Libertinus Serif вбудований у Typst
  // і має кирилицю, тож документ збереться на будь-якій машині.
  set text(
    font: ("Times New Roman", "Libertinus Serif"),
    size: 13pt,
    lang: "uk",
    region: "UA",
    hyphenate: true,
  )
  set par(justify: true, first-line-indent: (amount: 1.25cm, all: true), leading: 0.75em, spacing: 0.75em)

  // ---------- Заголовки ----------
  set heading(numbering: "1.1.")
  show heading: set text(size: 13pt)
  show heading: it => block(above: 1.4em, below: 1em, it)
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    set align(center)
    set text(size: 14pt)
    upper(it)
  }

  // ---------- Код ----------
  // DejaVu Sans Mono вбудований у Typst і підтримує кирилицю.
  show raw: set text(font: ("DejaVu Sans Mono", "Fira Code"), size: 10pt)
  // Блоки коду: сіре тло, рамка, номери рядків.
  show raw.where(block: true): it => {
    set par(justify: false, first-line-indent: 0pt)
    set align(left)
    block(
      width: 100%,
      fill: luma(247),
      stroke: 0.5pt + luma(200),
      radius: 3pt,
      inset: 8pt,
      grid(
        columns: (auto, 1fr),
        column-gutter: 8pt,
        row-gutter: 0.65em,
        ..it
          .lines
          .map(l => (
            text(fill: luma(150), str(l.number)),
            l,
          ))
          .flatten()
      ),
    )
  }
  // Inline-код: легке тло.
  show raw.where(block: false): box.with(
    fill: luma(240),
    inset: (x: 3pt, y: 0pt),
    outset: (y: 3pt),
    radius: 2pt,
  )

  // ---------- Рисунки, таблиці, лістинги ----------
  show figure.where(kind: raw): set figure(supplement: [Лістинг])
  show figure.where(kind: raw): set figure.caption(position: top)
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.caption: set text(size: 12pt)
  set math.equation(numbering: "(1)")

  // ---------- Титульна сторінка ----------
  page(numbering: none, {
    set align(center)
    set par(first-line-indent: 0pt)
    if university != none { text(size: 14pt, upper(university)) }
    v(1fr)
    text(size: 18pt, weight: "bold", title)
    if subtitle != none {
      v(0.5em)
      text(size: 14pt, subtitle)
    }
    v(1fr)
    align(right, block(width: 50%, align(left, {
      [Виконав: #author]
      if supervisor != none {
        linebreak()
        [Керівник: #supervisor]
      }
    })))
    v(1fr)
    [#city — #year]
  })

  // ---------- Зміст ----------
  {
    show heading: it => align(center, upper(it))
    outline(title: [Зміст], indent: auto)
  }

  body
}
