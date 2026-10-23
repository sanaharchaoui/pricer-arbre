import svg


# convertit un entier en hexadécimal avec au moins deux caractères, ex : 15 -> '0F' et 0 -> '00'
def to_hexa_two_char(number: int | float) -> str:
    if number > 15:
        return (hex(int(number)))[2:]
    else:
        return '0' + (hex(int(number)))[2:]


# cette classe aide à faire une image vectorielle svg et à l'enregistrer dans un fichier
class VectorImage:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.image = svg.SVG(width=width, height=height, elements=[])

    # ajoute un élément : affiché au-dessus des formes déjà dans l'image
    def add_end(self, element: svg.Element):
        self.image.elements += [element]

    # ajoute un élément : affiché en dessous des formes déjà dans l'image
    def add_beginning(self, element: svg.Element):
        self.image.elements.insert(0, element)

    # fait un cercle
    @staticmethod
    def circle(center_x: int, center_y: int, radius: int, stroke_color: str, fill_color: str,
               stroke_width: int, opacity: float = 1.0) -> svg.Circle:
        return svg.Circle(cx=center_x, cy=center_y, r=radius, stroke=stroke_color, fill=fill_color,
                          stroke_width=stroke_width, opacity=opacity)

    # fait un rectangle
    @staticmethod
    def rectangle(top_left_x: int, top_left_y: int, width: int, height: int, stroke_color: str,
                  fill_color: str, stroke_width: int, rounded_corner: int = 0,
                  opacity: float = 1.0) -> svg.Rect:
        return svg.Rect(x=top_left_x, y=top_left_y, width=width, height=height, rx=rounded_corner,
                        stroke=stroke_color, fill=fill_color, stroke_width=stroke_width, opacity=opacity)

    # fait un segment
    @staticmethod
    def segment(p1_x: int, p1_y: int, p2_x: int, p2_y: int, stroke_color: str, stroke_width: int,
                opacity: float = 1.0) -> svg.Line:
        return svg.Line(x1=p1_x, y1=p1_y, x2=p2_x, y2=p2_y, stroke=stroke_color,
                        stroke_width=stroke_width, opacity=opacity)

    # fait un texte centré sur x (pas dans le mémo du cours, ajouté pour les noeuds)
    @staticmethod
    def text(x: int, y: int, content: str, color: str, font_size: int = 11) -> svg.Text:
        return svg.Text(x=x, y=y, text=content, fill=color, font_size=font_size, text_anchor='middle')

    # enregistre le fichier
    def save(self, filename: str):
        with open(filename, 'w') as f:
            f.write(self.image.as_str())
