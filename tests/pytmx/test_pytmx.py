import unittest

import pytmx
import base64
import gzip
import zlib
import struct
from pytmx import (
    TiledElement,
    TiledMap,
    convert_to_bool,
    TileFlags,
    decode_gid,
    unpack_gids,
)

# Tiled gid flags
GID_TRANS_FLIPX = 1 << 31
GID_TRANS_FLIPY = 1 << 30
GID_TRANS_ROT = 1 << 29
GID_MASK = GID_TRANS_FLIPX | GID_TRANS_FLIPY | GID_TRANS_ROT


class TestConvertToBool(unittest.TestCase):
    def test_string_string_true(self) -> None:
        self.assertTrue(convert_to_bool("1"))
        self.assertTrue(convert_to_bool("y"))
        self.assertTrue(convert_to_bool("Y"))
        self.assertTrue(convert_to_bool("t"))
        self.assertTrue(convert_to_bool("T"))
        self.assertTrue(convert_to_bool("yes"))
        self.assertTrue(convert_to_bool("Yes"))
        self.assertTrue(convert_to_bool("YES"))
        self.assertTrue(convert_to_bool("true"))
        self.assertTrue(convert_to_bool("True"))
        self.assertTrue(convert_to_bool("TRUE"))

    def test_string_string_false(self) -> None:
        self.assertFalse(convert_to_bool("0"))
        self.assertFalse(convert_to_bool("n"))
        self.assertFalse(convert_to_bool("N"))
        self.assertFalse(convert_to_bool("f"))
        self.assertFalse(convert_to_bool("F"))
        self.assertFalse(convert_to_bool("no"))
        self.assertFalse(convert_to_bool("No"))
        self.assertFalse(convert_to_bool("NO"))
        self.assertFalse(convert_to_bool("false"))
        self.assertFalse(convert_to_bool("False"))
        self.assertFalse(convert_to_bool("FALSE"))

    def test_string_number_true(self) -> None:
        self.assertTrue(convert_to_bool(1))
        self.assertTrue(convert_to_bool(1.0))

    def test_string_number_false(self) -> None:
        self.assertFalse(convert_to_bool(0))
        self.assertFalse(convert_to_bool(0.0))
        self.assertFalse(convert_to_bool(-1))
        self.assertFalse(convert_to_bool(-1.1))

    def test_string_bool_true(self) -> None:
        self.assertTrue(convert_to_bool(True))

    def test_string_bool_false(self) -> None:
        self.assertFalse(convert_to_bool(False))

    def test_string_bool_none(self) -> None:
        self.assertFalse(convert_to_bool(None))

    def test_string_bool_empty(self) -> None:
        self.assertFalse(convert_to_bool(""))

    def test_string_bool_whitespace_only(self) -> None:
        self.assertFalse(convert_to_bool(" "))

    def test_non_boolean_string_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            convert_to_bool("garbage")

    def test_non_boolean_number_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            convert_to_bool("200")

    def test_edge_cases(self):
        # Whitespace
        self.assertTrue(convert_to_bool("  t  "))
        self.assertFalse(convert_to_bool("  f  "))

        # Numeric edge cases
        self.assertTrue(convert_to_bool(1e-10))  # Very small positive number
        self.assertFalse(convert_to_bool(-1e-10))  # Very small negative number


class TiledMapTest(unittest.TestCase):
    filename = "tests/resources/test01.tmx"

    def setUp(self) -> None:
        self.m = pytmx.TiledMap(self.filename)

    def test_build_rects(self) -> None:
        try:
            from pytmx import util_pygame

            rects = util_pygame.build_rects(self.m, "Grass and Water", "tileset", None)
            self.assertEqual(rects[0], [0, 0, 240, 240])
            rects = util_pygame.build_rects(self.m, "Grass and Water", "tileset", 18)
            self.assertNotEqual(0, len(rects))
        except ImportError:
            pass

    def test_get_tile_image(self) -> None:
        image = self.m.get_tile_image(0, 0, 0)

    def test_get_tile_image_by_gid(self) -> None:
        image = self.m.get_tile_image_by_gid(0)
        self.assertIsNone(image)

        image = self.m.get_tile_image_by_gid(1)
        self.assertIsNotNone(image)

    def test_reserved_names_check_disabled_with_option(self) -> None:
        pytmx.TiledElement.allow_duplicate_names = False
        pytmx.TiledMap(allow_duplicate_names=True)
        self.assertTrue(pytmx.TiledElement.allow_duplicate_names)

    def test_map_width_height_is_int(self) -> None:
        self.assertIsInstance(self.m.width, int)
        self.assertIsInstance(self.m.height, int)

    def test_layer_width_height_is_int(self) -> None:
        self.assertIsInstance(self.m.layers[0].width, int)
        self.assertIsInstance(self.m.layers[0].height, int)

    def test_properties_are_converted_to_builtin_types(self) -> None:
        self.assertIsInstance(self.m.properties["test_bool"], bool)
        self.assertIsInstance(self.m.properties["test_color"], str)
        self.assertIsInstance(self.m.properties["test_file"], str)
        self.assertIsInstance(self.m.properties["test_float"], float)
        self.assertIsInstance(self.m.properties["test_int"], int)
        self.assertIsInstance(self.m.properties["test_string"], str)

    def test_properties_are_converted_to_correct_values(self) -> None:
        self.assertFalse(self.m.properties["test_bool"])
        self.assertTrue(self.m.properties["test_bool_true"])

    def test_pixels_to_tile_pos(self) -> None:
        self.assertEqual(self.m.pixels_to_tile_pos((0, 33)), (0, 2))
        self.assertEqual(self.m.pixels_to_tile_pos((33, 0)), (2, 0))
        self.assertEqual(self.m.pixels_to_tile_pos((0, 0)), (0, 0))
        self.assertEqual(self.m.pixels_to_tile_pos((65, 86)), (4, 5))


class TiledElementTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.element = TiledElement()

    def test_from_xml_string_should_raise_on_TiledElement(self) -> None:
        with self.assertRaises(AttributeError):
            TiledElement.from_xml_string("<element></element>")

    def test_contains_reserved_property_name(self) -> None:
        """Reserved names are checked from any attributes in the instance
        after it is created.  Instance attributes are defaults from the
        specification.  We check that new properties are not named same
        as existing attributes.
        """
        self.element.name = "foo"
        items = {"name": None}
        result = self.element._contains_invalid_property_name(items.items())
        self.assertTrue(result)

    def test_not_contains_reserved_property_name(self) -> None:
        """Reserved names are checked from any attributes in the instance
        after it is created.  Instance attributes are defaults from the
        specification.  We check that new properties are not named same
        as existing attributes.
        """
        items = {"name": None}
        result = self.element._contains_invalid_property_name(items.items())
        self.assertFalse(result)

    def test_reserved_names_check_disabled_with_option(self) -> None:
        """Reserved names are checked from any attributes in the instance
        after it is created.  Instance attributes are defaults from the
        specification.  We check that new properties are not named same
        as existing attributes.

        Check that passing an option will disable the check
        """
        pytmx.TiledElement.allow_duplicate_names = True
        self.element.name = "foo"
        items = {"name": None}
        result = self.element._contains_invalid_property_name(items.items())
        self.assertFalse(result)

    def test_repr(self) -> None:
        self.element.name = "foo"
        self.assertEqual('<TiledElement: "foo">', self.element.__repr__())


class TestDecodeGid(unittest.TestCase):
    def test_no_flags(self):
        raw_gid = 100
        expected_gid, expected_flags = 100, TileFlags(False, False, False)
        self.assertEqual(decode_gid(raw_gid), (expected_gid, expected_flags))

    def test_individual_flags(self):
        # Test for each flag individually
        test_cases = [
            (GID_TRANS_FLIPX + 1, 1, TileFlags(True, False, False)),
            (GID_TRANS_FLIPY + 1, 1, TileFlags(False, True, False)),
            (GID_TRANS_ROT + 1, 1, TileFlags(False, False, True)),
        ]
        for raw_gid, expected_gid, expected_flags in test_cases:
            self.assertEqual(decode_gid(raw_gid), (expected_gid, expected_flags))

    def test_combinations_of_flags(self):
        # Test combinations of flags
        test_cases = [
            (GID_TRANS_FLIPX + GID_TRANS_FLIPY + 1, 1, TileFlags(True, True, False)),
            (GID_TRANS_FLIPX + GID_TRANS_ROT + 1, 1, TileFlags(True, False, True)),
            (GID_TRANS_FLIPY + GID_TRANS_ROT + 1, 1, TileFlags(False, True, True)),
            (
                GID_TRANS_FLIPX + GID_TRANS_FLIPY + GID_TRANS_ROT + 1,
                1,
                TileFlags(True, True, True),
            ),
        ]
        for raw_gid, expected_gid, expected_flags in test_cases:
            self.assertEqual(decode_gid(raw_gid), (expected_gid, expected_flags))

    def test_edge_cases(self):
        # Maximum GID
        max_gid = 2**29 -1
        self.assertEqual(decode_gid(max_gid), (max_gid & ~GID_MASK, TileFlags(False, False, False)))

        # Minimum GID
        min_gid = 0
        self.assertEqual(decode_gid(min_gid), (min_gid, TileFlags(False, False, False)))

        # GID with all flags set
        gid_all_flags = GID_TRANS_FLIPX + GID_TRANS_FLIPY + GID_TRANS_ROT + 1
        self.assertEqual(decode_gid(gid_all_flags), (1, TileFlags(True, True, True)))

        # GID with flags in different orders
        test_cases = [
            (GID_TRANS_FLIPX + GID_TRANS_FLIPY + 1, 1, TileFlags(True, True, False)),
            (GID_TRANS_FLIPY + GID_TRANS_FLIPX + 1, 1, TileFlags(True, True, False)),
            (GID_TRANS_FLIPX + GID_TRANS_ROT + 1, 1, TileFlags(True, False, True)),
            (GID_TRANS_ROT + GID_TRANS_FLIPX + 1, 1, TileFlags(True, False, True)),
        ]
        for raw_gid, expected_gid, expected_flags in test_cases:
            self.assertEqual(decode_gid(raw_gid), (expected_gid, expected_flags))


class TestRegisterGid(unittest.TestCase):
    def setUp(self):
        self.tmx_map = TiledMap()

    def test_register_gid_with_valid_tiled_gid(self):
        gid = self.tmx_map.register_gid(123)
        self.assertIsNotNone(gid)

    def test_register_gid_with_flags(self):
        flags = TileFlags(1, 0, 1)
        gid = self.tmx_map.register_gid(456, flags)
        self.assertIsNotNone(gid)

    def test_register_gid_zero(self):
        gid = self.tmx_map.register_gid(0)
        self.assertEqual(gid, 0)

    def test_register_gid_max_gid(self):
        max_gid = self.tmx_map.maxgid
        self.tmx_map.register_gid(max_gid)
        self.assertEqual(self.tmx_map.maxgid, max_gid + 1)

    def test_register_gid_duplicate_gid(self):
        gid1 = self.tmx_map.register_gid(123)
        gid2 = self.tmx_map.register_gid(123)
        self.assertEqual(gid1, gid2)

    def test_register_gid_duplicate_gid_different_flags(self):
        gid1 = self.tmx_map.register_gid(123, TileFlags(1, 0, 0))
        gid2 = self.tmx_map.register_gid(123, TileFlags(0, 1, 0))
        self.assertNotEqual(gid1, gid2)

    def test_register_gid_empty_flags(self):
        gid = self.tmx_map.register_gid(123, TileFlags(0, 0, 0))
        self.assertIsNotNone(gid)

    def test_register_gid_all_flags_set(self):
        gid = self.tmx_map.register_gid(123, TileFlags(1, 1, 1))
        self.assertIsNotNone(gid)

    def test_register_gid_repeated_registration(self):
        gid1 = self.tmx_map.register_gid(123)
        gid2 = self.tmx_map.register_gid(123)
        self.assertEqual(gid1, gid2)


class TestUnpackGids(unittest.TestCase):
    def test_base64_no_compression(self):
        gids = [123, 456, 789]
        data = struct.pack("<LLL", *gids)
        text = base64.b64encode(data).decode("utf-8")
        result = unpack_gids(text, encoding="base64")
        self.assertEqual(result, gids)

    def test_base64_gzip_compression(self):
        gids = [123, 456, 789]
        data = struct.pack("<LLL", *gids)
        compressed_data = gzip.compress(data)
        text = base64.b64encode(compressed_data).decode("utf-8")
        result = unpack_gids(text, encoding="base64", compression="gzip")
        self.assertEqual(result, gids)

    def test_base64_zlib_compression(self):
        gids = [123, 456, 789]
        data = struct.pack("<LLL", *gids)
        compressed_data = zlib.compress(data)
        text = base64.b64encode(compressed_data).decode("utf-8")
        result = unpack_gids(text, encoding="base64", compression="zlib")
        self.assertEqual(result, gids)

    def test_base64_unsupported_compression(self):
        text = "some_base64_data"
        with self.assertRaises(ValueError):
            unpack_gids(text, encoding="base64", compression="unsupported")

    def test_csv(self):
        gids = [123, 456, 789]
        text = ",".join(map(str, gids))
        result = unpack_gids(text, encoding="csv")
        self.assertEqual(result, gids)

    def test_unsupported_encoding(self):
        text = "some_data"
        with self.assertRaises(ValueError):
            unpack_gids(text, encoding="unsupported")
