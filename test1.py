from manim import *

class DeriveCircleArea(Scene):
    def construct(self):
        # Step 1: Draw a circle
        circle = Circle(radius=2, color=BLUE).set_opacity(0.5)
        self.add(circle)

        # Step 2: Label the circle
        label = Text("Circle").next_to(circle, RIGHT)
        self.add(label)

        # Step 3: Define the formula for the area of a circle
        formula = MathTex(r"A = \pi r^2").shift(UP)

        # Step 4: Derive the formula
        self.play(Write(formula))

        # Step 5: Explain the derivation
        explanation = Text("The area of a circle is equal to pi times the radius squared.")
        explanation.shift(DOWN)
        self.play(Write(explanation))

        # Step 6: Visualize the formula
        self.play(ReplacementTransform(formula, MathTex(r"A = \pi (\frac{d}{2})^2").shift(UP)))

        # Step 7: Explain the final formula
        final_explanation = Text("Since the diameter is twice the radius, we can substitute d/2 for r.")
        final_explanation.shift(DOWN)
        self.play(Write(final_explanation))

        # Step 8: Show the final formula
        self.play(Write(MathTex(r"A = \frac{\pi d^2}{4}").shift(UP)))

        # Step 9: Wait for 2 seconds
        self.wait(2)