from manim import *

class DeriveCircleArea(Scene):
    def construct(self):
        # Title
        title = Text("Derivation of Circle Area Formula")
        self.play(Write(title))
        self.wait()

        # Introduction
        intro = Text("To find the area of a circle, we can use the following steps:")
        self.play(Write(intro))
        self.wait()

        # Step 1: Divide the circle into sectors
        step1 = Text("Step 1: Divide the circle into sectors")
        self.play(Write(step1))
        self.wait()
        circle = Circle(radius=1)
        self.play(FadeIn(circle))
        self.wait()
        sectors = VGroup(*[Circle(radius=1, color=GREEN) for _ in range(10)])
        for sector in sectors:
            self.play(FadeIn(sector))
            self.wait(0.5)
        self.wait()

        # Step 2: Find the area of each sector
        step2 = Text("Step 2: Find the area of each sector")
        self.play(Write(step2))
        self.wait()
        sector_angle = MathTex("\\theta")
        self.play(FadeIn(sector_angle))
        self.wait()
        sector_area = MathTex("\\frac{1}{2}r^2\\sin\\theta")
        self.play(FadeIn(sector_area))
        self.wait()

        # Step 3: Sum the areas of all sectors
        step3 = Text("Step 3: Sum the areas of all sectors")
        self.play(Write(step3))
        self.wait()
        total_area = MathTex("\\pi r^2")
        self.play(FadeIn(total_area))
        self.wait()

        # Conclusion
        conclusion = Text("Therefore, the area of a circle is \\pi r^2.")
        self.play(Write(conclusion))
        self.wait()

        # Final circle
        final_circle = Circle(radius=1, color=BLUE)
        self.play(FadeIn(final_circle))
        self.wait()