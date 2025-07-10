from manim import *

class PythagoreanTheoremProof(Scene):
    def construct(self):
        # Define side lengths for the generic triangle for visualization
        # Using specific values (e.g., 3-4-5) makes exact placement and visualization clearer.
        a_length = 3
        b_length = 4
        c_length = np.sqrt(a_length**2 + b_length**2) # This will be 5 for a=3, b=4

        # 1. Introduction: Theorem Statement
        theorem_title = Title("The Pythagorean Theorem").to_edge(UP)
        theorem_formula = MathTex("a^2 + b^2 = c^2").next_to(theorem_title, DOWN, buff=0.7)
        self.play(Write(theorem_title), FadeIn(theorem_formula, shift=UP))
        self.wait(1.5)
        self.play(FadeOut(theorem_title), FadeOut(theorem_formula))
        self.wait(0.5)

        # 2. Right Triangle Introduction
        # Define vertices for a right-angled triangle at the bottom-left corner
        # C (origin), A (b_length, 0), B (0, a_length)
        vertex_C_pos = ORIGIN
        vertex_A_pos = b_length * RIGHT
        vertex_B_pos = a_length * UP

        triangle = Polygon(
            vertex_C_pos,
            vertex_A_pos,
            vertex_B_pos,
            color=BLUE, fill_opacity=0.3, stroke_width=4
        )

        # Labels for vertices (A, B, C)
        label_A_tri = Tex("A").next_to(vertex_A_pos, DR, buff=0.1)
        label_B_tri = Tex("B").next_to(vertex_B_pos, UL, buff=0.1)
        label_C_tri = Tex("C").next_to(vertex_C_pos, DL, buff=0.1)
        
        # Labels for sides (a, b, c)
        side_b_line = Line(vertex_C_pos, vertex_A_pos)
        label_b_tri = MathTex("b").next_to(side_b_line, DOWN, buff=0.1)

        side_a_line = Line(vertex_C_pos, vertex_B_pos)
        label_a_tri = MathTex("a").next_to(side_a_line, LEFT, buff=0.1)

        hypotenuse_line = Line(vertex_A_pos, vertex_B_pos)
        label_c_tri = MathTex("c").next_to(hypotenuse_line, UP + RIGHT, buff=0.1).rotate(hypotenuse_line.get_angle())

        # Right angle indication
        right_angle_mark = RightAngle(side_b_line, side_a_line, length=0.4, quadrant=(-1,-1), color=YELLOW)

        # Group all triangle elements for easy manipulation
        initial_triangle_group = VGroup(
            triangle, label_A_tri, label_B_tri, label_C_tri,
            label_a_tri, label_b_tri, label_c_tri, right_angle_mark
        ).scale(0.8).to_corner(UL, buff=0.7) # Position it in the upper-left corner

        self.play(Create(initial_triangle_group), run_time=2)
        self.wait(1)

        # 3. The Proof Setup: Two Large Squares of side (a+b)
        total_side = a_length + b_length # This will be 3+4=7
        main_square_left = Square(side_length=total_side, color=WHITE, stroke_opacity=1).shift(LEFT * 3.5)
        main_square_right = Square(side_length=total_side, color=WHITE, stroke_opacity=1).shift(RIGHT * 3.5)

        self.play(
            Create(main_square_left),
            Create(main_square_right),
            initial_triangle_group.animate.to_corner(UL).scale(0.7) # Shrink and move the original triangle out of the way
        )
        self.wait(1)

        # Labels for the side lengths of the large squares
        label_left_side_len = MathTex(f"a+b").next_to(main_square_left.get_left(), LEFT, buff=0.1)
        label_bottom_left_side_len = MathTex(f"a+b").next_to(main_square_left.get_bottom(), DOWN, buff=0.1)
        
        label_right_side_len = MathTex(f"a+b").next_to(main_square_right.get_right(), RIGHT, buff=0.1)
        label_bottom_right_side_len = MathTex(f"a+b").next_to(main_square_right.get_bottom(), DOWN, buff=0.1)

        self.play(
            Write(label_left_side_len), Write(label_bottom_left_side_len),
            Write(label_right_side_len), Write(label_bottom_right_side_len)
        )
        self.wait(1)

        # 4. Dissection of Left Square: (a^2 + b^2 + 2ab)
        # It consists of a square of side 'a', a square of side 'b', and two rectangles of a x b.
        
        sq_a_in_left = Square(side_length=a_length, color=RED, fill_opacity=0.7).align_to(main_square_left, DL)
        sq_b_in_left = Square(side_length=b_length, color=BLUE, fill_opacity=0.7).align_to(main_square_left, UR)

        rect_ul_in_left = Rectangle(width=b_length, height=a_length, color=GREEN, fill_opacity=0.5)
        rect_ul_in_left.align_to(main_square_left, UL).shift(RIGHT * a_length)

        rect_dr_in_left = Rectangle(width=a_length, height=b_length, color=GREEN, fill_opacity=0.5)
        rect_dr_in_left.align_to(main_square_left, DR).shift(UP * a_length)

        # Labels for the areas
        label_a2_in_left = MathTex("a^2").move_to(sq_a_in_left)
        label_b2_in_left = MathTex("b^2").move_to(sq_b_in_left)
        label_ab1_in_left = MathTex("ab").move_to(rect_ul_in_left)
        label_ab2_in_left = MathTex("ab").move_to(rect_dr_in_left)

        self.play(
            FadeIn(sq_a_in_left), FadeIn(sq_b_in_left),
            FadeIn(rect_ul_in_left), FadeIn(rect_dr_in_left)
        )
        self.play(
            Write(label_a2_in_left), Write(label_b2_in_left),
            Write(label_ab1_in_left), Write(label_ab2_in_left)
        )
        self.wait(1.5)

        # 5. Dissection of Right Square: (c^2 + 4 * (1/2 * ab))
        # It consists of a central square of side 'c' (rotated) and four identical right triangles.
        
        # Get corner points of the right large square for precise triangle placement
        s_right_dl = main_square_right.get_corner(DL)
        s_right_dr = main_square_right.get_corner(DR)
        s_right_ul = main_square_right.get_corner(UL)
        s_right_ur = main_square_right.get_corner(UR)

        # Vertices of the inner square (c^2) based on the standard proof diagram
        P_c1 = s_right_dl + b_length * RIGHT  # Bottom vertex
        P_c2 = s_right_dr + a_length * UP     # Right vertex
        P_c3 = s_right_ur + b_length * LEFT   # Top vertex
        P_c4 = s_right_ul + a_length * DOWN   # Left vertex

        sq_c_in_right = Polygon(P_c1, P_c2, P_c3, P_c4, color=PURPLE, fill_opacity=0.7, stroke_width=2)
        label_c2_in_right = MathTex("c^2").move_to(sq_c_in_right)

        # Four triangles filling the corners
        triangles_in_right_group = VGroup()
        triangles_in_right_group.add(Polygon(s_right_dl, P_c1, P_c4, color=GREEN, fill_opacity=0.5)) # Bottom-left triangle
        triangles_in_right_group.add(Polygon(P_c1, s_right_dr, P_c2, color=GREEN, fill_opacity=0.5)) # Bottom-right triangle
        triangles_in_right_group.add(Polygon(P_c2, s_right_ur, P_c3, color=GREEN, fill_opacity=0.5)) # Top-right triangle
        triangles_in_right_group.add(Polygon(P_c3, s_right_ul, P_c4, color=GREEN, fill_opacity=0.5)) # Top-left triangle

        # Labels for the areas of each triangle
        label_tri1_in_right = MathTex("\\frac{1}{2}ab").move_to(triangles_in_right_group[0].get_center())
        label_tri2_in_right = MathTex("\\frac{1}{2}ab").move_to(triangles_in_right_group[1].get_center())
        label_tri3_in_right = MathTex("\\frac{1}{2}ab").move_to(triangles_in_right_group[2].get_center())
        label_tri4_in_right = MathTex("\\frac{1}{2}ab").move_to(triangles_in_right_group[3].get_center())

        self.play(
            FadeIn(sq_c_in_right),
            FadeIn(triangles_in_right_group)
        )
        self.play(
            Write(label_c2_in_right),
            Write(label_tri1_in_right),
            Write(label_tri2_in_right),
            Write(label_tri3_in_right),
            Write(label_tri4_in_right)
        )
        self.wait(1.5)

        # 6. Comparison and Conclusion
        self.play(
            FadeOut(label_left_side_len), FadeOut(label_bottom_left_side_len),
            FadeOut(label_right_side_len), FadeOut(label_bottom_right_side_len)
        )
        
        # Show area expressions for both large squares
        area_eq_left = MathTex("Area = a^2 + b^2 + 2ab").next_to(main_square_left, UP, buff=0.5)
        area_eq_right = MathTex("Area = c^2 + 4 \\times \\frac{1}{2}ab").next_to(main_square_right, UP, buff=0.5)
        
        self.play(Write(area_eq_left), Write(area_eq_right))
        self.wait(1)

        # Simplify the right side's area expression
        area_eq_right_simplified = MathTex("Area = c^2 + 2ab").move_to(area_eq_right.get_center())
        self.play(Transform(area_eq_right, area_eq_right_simplified))
        self.wait(1)

        # Indicate that the two large squares have equal total areas
        equal_sign_between_squares = MathTex("=").move_to(VGroup(main_square_left, main_square_right).get_center())
        self.play(Create(equal_sign_between_squares))
        self.play(Indicate(main_square_left), Indicate(main_square_right), Indicate(equal_sign_between_squares))
        self.wait(1)

        # Highlight the common areas (2ab) in both squares
        common_areas_left = VGroup(rect_ul_in_left, label_ab1_in_left, rect_dr_in_left, label_ab2_in_left)
        common_areas_right = VGroup(triangles_in_right_group, label_tri1_in_right, label_tri2_in_right, label_tri3_in_right, label_tri4_in_right)

        self.play(
            Indicate(common_areas_left, color=YELLOW),
            Indicate(common_areas_right, color=YELLOW)
        )
        self.wait(1)

        # Fade out the common areas and transform the area expressions
        self.play(
            FadeOut(common_areas_left, target_mode="fade_out"), # Keep them slightly visible during transform
            FadeOut(common_areas_right, target_mode="fade_out"),
            FadeOut(equal_sign_between_squares),
            Transform(area_eq_left, MathTex("a^2 + b^2").move_to(area_eq_left.get_center())),
            Transform(area_eq_right, MathTex("c^2").move_to(area_eq_right.get_center()))
        )
        # Fully remove the common area mobjects after the fade out
        self.remove(*common_areas_left, *common_areas_right) 

        self.wait(1.5)

        # Arrange the remaining areas (a^2, b^2, c^2) and the final equation
        final_equal_sign = MathTex("=").move_to(ORIGIN)
        self.play(Create(final_equal_sign))
        self.wait(0.5)

        # Group remaining square components with their labels
        final_a2_group = VGroup(sq_a_in_left, label_a2_in_left)
        final_b2_group = VGroup(sq_b_in_left, label_b2_in_left)
        final_c2_group = VGroup(sq_c_in_right, label_c2_in_right)

        # Prepare for arrangement: a^2 + b^2 = c^2
        plus_final = MathTex("+")
        
        # Animate the remaining squares and labels moving towards their final positions
        self.play(
            final_a2_group.animate.next_to(final_equal_sign, LEFT, buff=3.0),
            final_b2_group.animate.next_to(final_equal_sign, LEFT, buff=1.0),
            plus_final.animate.next_to(final_a2_group, RIGHT, buff=0.5),
            final_c2_group.animate.next_to(final_equal_sign, RIGHT, buff=1.0),
            FadeOut(main_square_left), # Fade out the frames of the large squares
            FadeOut(main_square_right),
            FadeOut(area_eq_left), # Fade out the area text
            FadeOut(area_eq_right),
            FadeOut(initial_triangle_group) # Fade out the initial triangle group
        )
        self.add(plus_final) # Ensure plus sign is added to the scene

        # Arrange them precisely into the final formula layout
        final_elements = VGroup(
            final_a2_group,
            plus_final,
            final_b2_group,
            final_equal_sign,
            final_c2_group
        ).arrange(RIGHT, buff=0.5).move_to(ORIGIN)

        self.play(
            final_a2_group.animate.move_to(final_elements[0].get_center()),
            plus_final.animate.move_to(final_elements[1].get_center()),
            final_b2_group.animate.move_to(final_elements[2].get_center()),
            final_equal_sign.animate.move_to(final_elements[3].get_center()),
            final_c2_group.animate.move_to(final_elements[4].get_center()),
            run_time=1.5
        )
        self.wait(1.5)

        # Transform the visual components into the final theorem text
        final_theorem_text = MathTex("a^2 + b^2 = c^2").move_to(ORIGIN).scale(1.5)
        self.play(
            Transform(final_elements, final_theorem_text)
        )
        self.wait(2)
        
        self.play(FadeOut(final_theorem_text))
        self.wait(1)