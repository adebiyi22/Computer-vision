"""Builds the Module 3 assignment PDF report."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak,
    Table, TableStyle, ListFlowable, ListItem
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Mono", fontName="Courier", fontSize=9,
                           leading=12, backColor=colors.whitesmoke,
                           borderPadding=6, spaceAfter=8, spaceBefore=4))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceBefore=16))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=12))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], spaceAfter=8, leading=14))
styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8.5,
                           textColor=colors.grey))

GITHUB_LINK_PLACEHOLDER = "PASTE_YOUR_GITHUB_REPO_LINK_HERE"
WEBAPP_LINK_PLACEHOLDER = "PASTE_YOUR_DEPLOYED_STREAMLIT_APP_LINK_HERE"

story = []

# ---- Title ----
story.append(Paragraph("CSc 8830: Computer Vision", styles["Title"]))
story.append(Paragraph("Module 3 Assignment", styles["Heading2"]))
story.append(Paragraph("Image Blurring: Spatial Filtering vs. Fourier-Domain Filtering",
                        styles["Heading3"]))
story.append(Spacer(1, 12))
story.append(Paragraph(f"<b>GitHub repository:</b> {GITHUB_LINK_PLACEHOLDER}", styles["Body"]))
story.append(Paragraph(f"<b>Deployed web application:</b> {WEBAPP_LINK_PLACEHOLDER}", styles["Body"]))
story.append(Spacer(1, 12))

# ---- Section 1: Assignment restatement ----
story.append(Paragraph("1. Assignment", styles["H1"]))
story.append(Paragraph(
    "<b>Implement:</b> Image blurring using a filtering approach.", styles["Body"]))
story.append(Paragraph(
    "<b>Theory:</b> Show that the outcome using spatial filters is the same as using "
    "the Fourier-domain equivalent of the filter. Show that convolution in space is "
    "the same as multiplication in the frequency (Fourier) domain. Use the "
    "implementation from above and experimentation for showing evidence/validation.",
    styles["Body"]))

# ---- Section 2: Implementation ----
story.append(Paragraph("2. Implementation", styles["H1"]))
story.append(Paragraph(
    "A Streamlit web application (<font face='Courier'>app.py</font>) implements image "
    "blurring two independent ways and compares them numerically:", styles["Body"]))
story.append(ListFlowable([
    ListItem(Paragraph("Constructs a normalized 2D Gaussian kernel of a chosen size "
                        "and standard deviation (&sigma;), selectable via a slider.", styles["Body"])),
    ListItem(Paragraph("<b>Spatial-domain filtering:</b> direct 2D convolution of the "
                        "image with the kernel using circular (\"wrap\") boundary "
                        "conditions.", styles["Body"])),
    ListItem(Paragraph("<b>Fourier-domain filtering:</b> the kernel is zero-padded to "
                        "the image size and circularly shifted so its center sits at "
                        "the origin, then both the image and kernel are transformed "
                        "with the 2D FFT, multiplied pointwise, and inverse-transformed.",
                        styles["Body"])),
    ListItem(Paragraph("The two results are compared pixel-by-pixel using Mean "
                        "Absolute Error (MAE), Root Mean Square Error (RMSE), and "
                        "maximum absolute difference.", styles["Body"])),
], bulletType="bullet"))

story.append(Paragraph("Core filtering functions:", styles["H2"]))
code = """def gaussian_kernel(size, sigma):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def spatial_filter(image, kernel):
    return convolve2d(image, kernel, mode="same", boundary="wrap")

def fourier_filter(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    padded = np.zeros_like(image, dtype=np.float64)
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    F = fft2(image)
    H = fft2(padded)
    return np.real(ifft2(F * H))"""
story.append(Paragraph(code.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles["Mono"]))
story.append(Paragraph(
    "Full commented source with a top-of-file README/usage docstring is in "
    "<font face='Courier'>app.py</font> in the GitHub repository linked above.",
    styles["Small"]))

story.append(PageBreak())

# ---- Section 3: Theory ----
story.append(Paragraph("3. Theory: The Convolution Theorem", styles["H1"]))
story.append(Paragraph(
    "<b>Claim:</b> For an image f(x,y) and filter h(x,y), spatial convolution "
    "g = f * h satisfies F{f * h} = F{f} &middot; F{h}, where F{&middot;} denotes the "
    "2D Fourier transform.", styles["Body"]))

story.append(Paragraph("3.1 Definitions", styles["H2"]))
story.append(Paragraph(
    "The 2D discrete Fourier transform (DFT) of an M&times;N image f(x,y) is:", styles["Body"]))
story.append(Paragraph(
    "F(u,v) = &Sigma;<sub>x=0..M-1</sub> &Sigma;<sub>y=0..N-1</sub> f(x,y) "
    "exp(&minus;j2&pi;(ux/M + vy/N))", styles["Mono"]))
story.append(Paragraph(
    "Circular convolution of f and h (h embedded into an M&times;N array) is:", styles["Body"]))
story.append(Paragraph(
    "g(x,y) = &Sigma;<sub>m</sub> &Sigma;<sub>n</sub> f(m,n) h(x&minus;m mod M, y&minus;n mod N)",
    styles["Mono"]))

story.append(Paragraph("3.2 Derivation", styles["H2"]))
story.append(Paragraph("Take the DFT of g and substitute its definition:", styles["Body"]))
story.append(Paragraph(
    "G(u,v) = &Sigma;<sub>x,y</sub> [ &Sigma;<sub>m,n</sub> f(m,n) h(x&minus;m,y&minus;n) ] "
    "exp(&minus;j2&pi;(ux/M+vy/N))", styles["Mono"]))
story.append(Paragraph(
    "Exchanging the (finite) summation order and substituting "
    "x&prime;=x&minus;m, y&prime;=y&minus;n (valid since h is treated as periodic, "
    "matching the DFT's implicit periodicity):", styles["Body"]))
story.append(Paragraph(
    "G(u,v) = &Sigma;<sub>m,n</sub> f(m,n) exp(&minus;j2&pi;(um/M+vn/N)) &middot; "
    "[ &Sigma;<sub>x&prime;,y&prime;</sub> h(x&prime;,y&prime;) "
    "exp(&minus;j2&pi;(ux&prime;/M+vy&prime;/N)) ]", styles["Mono"]))
story.append(Paragraph(
    "The bracketed term is exactly H(u,v). Substituting it back:", styles["Body"]))
story.append(Paragraph(
    "G(u,v) = H(u,v) &middot; &Sigma;<sub>m,n</sub> f(m,n) exp(&minus;j2&pi;(um/M+vn/N)) "
    "= H(u,v) &middot; F(u,v)", styles["Mono"]))
story.append(Paragraph(
    "<b>Therefore G(u,v) = F(u,v)&middot;H(u,v)</b>, i.e. the Fourier transform of the "
    "convolution equals the pointwise product of the individual transforms "
    "(the Convolution Theorem). Applying the inverse DFT to both sides gives "
    "g(x,y) = F<super>-1</super>{F(u,v)H(u,v)}, which is the second form the "
    "implementation evaluates.", styles["Body"]))

story.append(Paragraph("3.3 Why the implementation matches this exactly", styles["H2"]))
story.append(Paragraph(
    "The derivation assumes <b>circular</b> convolution, since the DFT treats a "
    "signal as periodic. The implementation is written to match this assumption "
    "in two places: (1) the spatial convolution uses "
    "<font face='Courier'>boundary=\"wrap\"</font> rather than zero-padded "
    "(linear) boundaries, and (2) the kernel is zero-padded to the image size "
    "and circularly shifted (<font face='Courier'>np.roll</font>) so its center "
    "sits at index (0,0), matching the origin convention used in the derivation "
    "above. Without step (2), the two results would still be numerically close "
    "but spatially offset by half the kernel width/height.", styles["Body"]))

story.append(PageBreak())

# ---- Section 4: Experimental validation ----
story.append(Paragraph("4. Experimental Validation", styles["H1"]))
story.append(Paragraph(
    "The implementation above was run on a 256&times;256 synthetic test image "
    "containing sharp edges (a rectangle, an ellipse, a triangle, and a striped "
    "region) to make blur effects clearly visible, at &sigma;=3.0 "
    "(19&times;19 kernel):", styles["Body"]))

img_w = 1.55 * inch
row1 = Table([
    [RLImage("figures/original.png", width=img_w, height=img_w),
     RLImage("figures/spatial.png", width=img_w, height=img_w),
     RLImage("figures/fourier.png", width=img_w, height=img_w),
     RLImage("figures/difference.png", width=img_w, height=img_w)],
    [Paragraph("Original", styles["Small"]), Paragraph("Spatial conv.", styles["Small"]),
     Paragraph("Fourier mult.", styles["Small"]), Paragraph("|difference| (normalized)", styles["Small"])],
], colWidths=[img_w+8]*4)
row1.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER")]))
story.append(row1)
story.append(Spacer(1, 10))
story.append(Paragraph(
    "The spatial and Fourier outputs are visually indistinguishable, and the "
    "difference map is uniform low-level noise at the limits of display "
    "precision &mdash; consistent with the theorem predicting zero real "
    "difference between the two computations.", styles["Body"]))

story.append(Paragraph("4.1 Numerical error across blur strengths", styles["H2"]))
story.append(Paragraph(
    "The same comparison was repeated for ten values of &sigma; (0.5 to 10) to "
    "confirm the equivalence holds regardless of blur strength or kernel size:",
    styles["Body"]))

data = [["Sigma", "Kernel", "MAE", "RMSE", "Max abs error"],
        ["0.5", "5x5", "1.00e-16", "1.34e-16", "6.94e-16"],
        ["1.0", "7x7", "1.02e-16", "1.39e-16", "6.66e-16"],
        ["1.5", "11x11", "1.02e-16", "1.28e-16", "7.08e-16"],
        ["2.0", "13x13", "1.25e-16", "1.65e-16", "7.77e-16"],
        ["3.0", "19x19", "1.48e-16", "2.09e-16", "1.11e-15"],
        ["4.0", "25x25", "1.13e-16", "1.53e-16", "7.77e-16"],
        ["5.0", "31x31", "1.47e-16", "2.19e-16", "1.33e-15"],
        ["6.0", "37x37", "1.44e-16", "1.88e-16", "1.11e-15"],
        ["8.0", "49x49", "2.44e-16", "3.46e-16", "1.55e-15"],
        ["10.0", "61x61", "2.95e-16", "4.53e-16", "2.11e-15"]]
t = Table(data, colWidths=[0.7*inch, 0.7*inch, 1.1*inch, 1.1*inch, 1.3*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#333333")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
]))
story.append(t)
story.append(Spacer(1, 10))

story.append(RLImage("figures/error_vs_sigma.png", width=5.2*inch, height=3.47*inch))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "All errors remain at IEEE-754 double-precision floating-point round-off "
    "(~1e-16), independent of blur strength. This is exactly the behavior "
    "predicted by the theorem: the spatial and Fourier results are the same "
    "function evaluated two different ways, so the only observed discrepancy is "
    "floating-point arithmetic error, not a real difference between the "
    "methods.", styles["Body"]))

story.append(Paragraph("5. Conclusion", styles["H1"]))
story.append(Paragraph(
    "The experimental results confirm the Convolution Theorem for this "
    "implementation: spatial-domain Gaussian blurring and its Fourier-domain "
    "equivalent (FFT &rarr; multiply &rarr; inverse FFT) produce numerically "
    "identical results (differences at the level of floating-point precision) "
    "across a wide range of blur strengths, validating both the mathematical "
    "derivation in Section 3 and the correctness of the implementation in "
    "Section 2.", styles["Body"]))

story.append(Paragraph("6. Demonstration Video", styles["H1"]))
story.append(Paragraph(
    "See the submitted screen recording for a live demonstration of the web "
    "application: uploading an image, adjusting &sigma;, and comparing the "
    "spatial and Fourier results with their MAE/RMSE/max-error statistics.",
    styles["Body"]))

doc = SimpleDocTemplate("Module_3_Report.pdf", pagesize=letter,
                         topMargin=0.7*inch, bottomMargin=0.7*inch,
                         leftMargin=0.8*inch, rightMargin=0.8*inch)
doc.build(story)
print("PDF built.")
