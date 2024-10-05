import sys
import os
import math
from os.path import split as pathsplit
from collections import namedtuple
from PIL import Image
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
from collections import namedtuple
from decimal import Decimal, getcontext

# Constants for Segments
PDS = int('0x14', 16)
ODS = int('0x15', 16)
PCS = int('0x16', 16)
WDS = int('0x17', 16)
END = int('0x80', 16)

# Frame rate mapping
FRAME_RATE_MAP = {
    16: "23.976",
    32: "24",
    48: "25",
    64: "29.97",
    96: "50",
    112: "59.94"
}

# Named tuple access for static PDS palettes
Palette = namedtuple('Palette', "Y Cr Cb Alpha")

class InvalidSegmentError(Exception):
    '''Raised when a segment does not match PGS specification'''

class PGSReader:
    def __init__(self, filepath):
        self.filedir, self.file = pathsplit(filepath)
        with open(filepath, 'rb') as f:
            self.bytes = f.read()
        self.frame_rates = set()  # Change to set to store unique frame rates

    def make_segment(self, bytes_):
        cls = SEGMENT_TYPE.get(bytes_[10], None)
        if cls is None:
            raise InvalidSegmentError(f"Unknown segment type: {hex(bytes_[10])}")
        return cls(bytes_)

    def iter_segments(self):
        bytes_ = self.bytes[:]
        while bytes_:
            size = 13 + int(bytes_[11:13].hex(), 16)
            segment = self.make_segment(bytes_[:size])
            
            if hasattr(segment, 'frame_rate'):
                frame_rate = FRAME_RATE_MAP.get(segment.frame_rate)
                if frame_rate is not None:
                    frame_rate = float(frame_rate)
                    if frame_rate not in self.frame_rates:
                        self.frame_rates.add(frame_rate)  # Store frame rates in a set
                        # print(f"Frame rate: {frame_rate:.3f}")  # Debugging info

            bytes_ = bytes_[size:]

        return self.frame_rates  # Return the set of frame rates

    def get_frame_rates(self):
        return self.frame_rates

    def iter_displaysets(self):
        ds = []
        self.iter_segments()  # Process segments to update frame rates
        bytes_ = self.bytes[:]
        while bytes_:
            size = 13 + int(bytes_[11:13].hex(), 16)
            segment = self.make_segment(bytes_[:size])
            
            ds.append(segment)
            if segment.type == 'END':
                yield DisplaySet(ds)
                ds = []
            
            bytes_ = bytes_[size:]

    @property
    def segments(self):
        if not hasattr(self, '_segments'):
            self._segments = list(self.iter_segments())
        return self._segments

    @property
    def displaysets(self):
        if not hasattr(self, '_displaysets'):
            self._displaysets = list(self.iter_displaysets())
        return self._displaysets

class BaseSegment:

    SEGMENT = {
        PDS: 'PDS',
        ODS: 'ODS',
        PCS: 'PCS',
        WDS: 'WDS',
        END: 'END'
    }

    def __init__(self, bytes_):
        self.bytes = bytes_
        if bytes_[:2] != b'PG':
            raise InvalidSegmentError
        self.pts = int(bytes_[2:6].hex(), base=16) / 90
        self.dts = int(bytes_[6:10].hex(), base=16) / 90
        self.type = self.SEGMENT[bytes_[10]]
        self.size = int(bytes_[11:13].hex(), base=16)
        self.data = bytes_[13:]

    def __len__(self):
        return self.size

    @property
    def presentation_timestamp(self): return self.pts

    @property
    def decoding_timestamp(self): return self.dts

    @property
    def segment_type(self): return self.type

class PresentationCompositionSegment(BaseSegment):

    class CompositionObject:

        def __init__(self, bytes_):
            self.bytes = bytes_
            self.object_id = int(bytes_[0:2].hex(), 16)
            self.window_id = bytes_[2]
            self.cropped = bool(bytes_[3])
            self.x_offset = int(bytes_[4:6].hex(), 16)
            self.y_offset = int(bytes_[6:8].hex(), 16)
            if self.cropped:
                self.crop_x_offset = int(bytes_[8:10].hex(), 16)
                self.crop_y_offset = int(bytes_[10:12].hex(), 16)
                self.crop_width = int(bytes_[12:14].hex(), 16)
                self.crop_height = int(bytes_[14:16].hex(), 16)

    STATE = {
        int('0x00', 16): 'Normal',
        int('0x40', 16): 'Acquisition Point',
        int('0x80', 16): 'Epoch Start'
    }

    def __init__(self, bytes_):
        BaseSegment.__init__(self, bytes_)
        self.width = int(self.data[0:2].hex(), 16)
        self.height = int(self.data[2:4].hex(), 16)
        self.frame_rate = self.data[4]
        self._num = int(self.data[5:7].hex(), 16)
        self._state = self.STATE[self.data[7]]
        self.palette_update = bool(self.data[8])
        self.palette_id = self.data[9]
        self._num_comps = self.data[10]

    @property
    def composition_number(self): return self._num

    @property
    def composition_state(self): return self._state

    @property
    def composition_objects(self):
        if not hasattr(self, '_composition_objects'):
            self._composition_objects = self.get_composition_objects()
            if len(self._composition_objects) != self._num_comps:
                print('Warning: Number of composition objects asserted '
                      'does not match the amount found.')
        return self._composition_objects

    def get_composition_objects(self):
        bytes_ = self.data[11:]
        comps = []
        while bytes_:
            length = 8*(1 + bool(bytes_[3]))
            comps.append(self.CompositionObject(bytes_[:length]))
            bytes_ = bytes_[length:]
        return comps

class WindowDefinitionSegment(BaseSegment):

    def __init__(self, bytes_):
        BaseSegment.__init__(self, bytes_)
        self.num_windows = self.data[0]
        self.window_id = self.data[1]
        self.x_offset = int(self.data[2:4].hex(), 16)
        self.y_offset = int(self.data[4:6].hex(), 16)
        self.width = int(self.data[6:8].hex(), 16)
        self.height = int(self.data[8:10].hex(), 16)

class PaletteDefinitionSegment(BaseSegment):

    def __init__(self, bytes_):
        BaseSegment.__init__(self, bytes_)
        self.palette_id = self.data[0]
        self.version = self.data[1]
        self.palette = [Palette(0, 0, 0, 0)]*256
        # Slice from byte 2 til end of segment. Divide by 5 to determine number of palette entries
        # Iterate entries. Explode the 5 bytes into namedtuple Palette. Must be exploded
        for entry in range(len(self.data[2:])//5):
            i = 2 + entry*5
            self.palette[self.data[i]] = Palette(*self.data[i+1:i+5])

class ObjectDefinitionSegment(BaseSegment):

    SEQUENCE = {
        int('0x40', 16): 'Last',
        int('0x80', 16): 'First',
        int('0xc0', 16): 'First and last'
    }

    def __init__(self, bytes_):
        BaseSegment.__init__(self, bytes_)
        self.id = int(self.data[0:2].hex(), 16)
        self.version = self.data[2]
        self.in_sequence = self.SEQUENCE[self.data[3]]
        self.data_len = int(self.data[4:7].hex(), 16)
        self.width = int(self.data[7:9].hex(), 16)
        self.height = int(self.data[9:11].hex(), 16)
        self.img_data = self.data[11:]
        if len(self.img_data) != self.data_len - 4:
            print('Warning: Image data length asserted does not match the '
                  'length found.')

class EndSegment(BaseSegment):

    @property
    def is_end(self): return True

SEGMENT_TYPE = {
    PDS: PaletteDefinitionSegment,
    ODS: ObjectDefinitionSegment,
    PCS: PresentationCompositionSegment,
    WDS: WindowDefinitionSegment,
    END: EndSegment
}

class DisplaySet:

    def __init__(self, segments):
        self.segments = segments
        self.segment_types = [s.type for s in segments]
        self.has_image = 'ODS' in self.segment_types

def segment_by_type_getter(type_):
    def f(self):
        return [s for s in self.segments if s.type == type_]
    return f

for type_ in BaseSegment.SEGMENT.values():
    setattr(DisplaySet, type_.lower(), property(segment_by_type_getter(type_)))

def read_rle_bytes(ods_bytes):
    pixels = []
    line_builder = []
    i = 0
    while i < len(ods_bytes):
        if ods_bytes[i]:
            incr = 1
            color = ods_bytes[i]
            length = 1
        else:
            check = ods_bytes[i+1]
            if check == 0:
                incr = 2
                color = 0
                length = 0
                pixels.append(line_builder)
                line_builder = []
            elif check < 64:
                incr = 2
                color = 0
                length = check
            elif check < 128:
                incr = 3
                color = 0
                length = ((check - 64) << 8) + ods_bytes[i + 2]
            elif check < 192:
                incr = 3
                color = ods_bytes[i + 2]
                length = check - 128
            elif check < 256:
                incr = 4
                color = ods_bytes[i + 3]
                length = ((check - 192) << 8) + ods_bytes[i + 2]
        line_builder.extend([color]*length)
        i += incr
    if line_builder:
        pixels.append(line_builder)
    return pixels

def create_image(width, height, palette, pixels):
    if sum(len(row) for row in pixels) != width * height:
        raise ValueError("The number of pixels does not match the image dimensions")
    
    # 创建 RGBA 图像
    img = Image.new("RGBA", (width, height))
    
    # 填充像素（将YCbCr转换为RGB并设置到图像中）
    for y, row in enumerate(pixels):
        for x, pixel in enumerate(row):
            y_val, cr, cb, alpha = palette[pixel]
            r, g, b = ycbcr_to_rgb(y_val, cr, cb)
            img.putpixel((x, y), (r, g, b, alpha))
    
    # 获取 alpha 通道
    alpha = img.split()[3]
    
    # 创建一个透明背景的全透明图像
    background = Image.new('RGBA', img.size, (0, 0, 0, 0))
    
    # 将原始图像粘贴到透明背景上，保留透明区域
    background.paste(img, mask=alpha)
    
    # 将图像转换为 Indexed-8（调色板模式）
    indexed_image = background.convert('P', palette=Image.ADAPTIVE, colors=256)
    
    # 返回 Indexed-8 调色板模式的图像对象
    return indexed_image

def ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return int(r), int(g), int(b)

def create_xml_script(events, output_file, frame_rate):
    header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    bdn_open = '<BDN Version="0.93" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n' \
               'xsi:noNamespaceSchemaLocation="BD-03-006-0093b BDN File Format.xsd">\n'
    
    # Calculate timecodes for XML
    if events:
        first_event_in_tc = events[0]['start']
        last_event_out_tc = events[-1]['end']
        content_in_tc = first_event_in_tc
        content_out_tc = last_event_out_tc
    else:
        first_event_in_tc = "00:00:00:00"
        last_event_out_tc = "00:00:00:00"
        content_in_tc = "00:00:00:00"
        content_out_tc = "00:00:00:00"
        
    description = '<Description>\n' \
                  '<Name Title="" Content=""/>\n' \
                  '<Language Code="zho"/>\n' \
                  '<Format VideoFormat="1080p" FrameRate="{frame_rate}" DropFrame="False"/>\n' \
                  '<Events LastEventOutTC="{last_event_out_tc}" FirstEventInTC="{first_event_in_tc}"\n' \
                  'ContentInTC="{content_in_tc}" ContentOutTC="{content_out_tc}" NumberofEvents="{num_events}" Type="Graphic"/>\n' \
                  '</Description>\n'
    
    events_open = '<Events>\n'
    events_close = '</Events>\n'
    bdn_close = '</BDN>'

    events_str = ''
    for event in events:
        events_str += '<Event Forced="False" InTC="{start}" OutTC="{end}">\n' \
                      '  <Graphic Width="{width}" Height="{height}" X="{x}" Y="{y}">{image}</Graphic>\n' \
                      '</Event>\n'.format(
            start=event['start'],
            end=event['end'],
            width=event['width'],
            height=event['height'],
            x=event['x'],
            y=event['y'],
            image=event['image']
        )

    # Ensure proper formatting
    xml_content = (header +
                   bdn_open +
                   description.format(
                       frame_rate=f"{frame_rate:.3f}" if frame_rate % 1 != 0 else f"{int(frame_rate)}",
                       last_event_out_tc=last_event_out_tc,
                       first_event_in_tc=first_event_in_tc,
                       content_in_tc=content_in_tc,
                       content_out_tc=content_out_tc,
                       num_events=len(events)
                   ) +
                   events_open +
                   events_str +
                   events_close +
                   bdn_close)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(xml_content)

def format_timestamp(pts):
    # Assuming pts is in seconds, convert it to milliseconds
    return int(pts)

# 设置 decimal 精度
getcontext().prec = 28  # 你可以根据需要调整精度

def to_timecode_milliseconds(milliseconds):
    milliseconds = int(milliseconds)
    ms = milliseconds % 1000
    seconds = (milliseconds // 1000) % 60
    minutes = (milliseconds // (1000 * 60)) % 60
    hours = (milliseconds // (1000 * 60 * 60)) % 24
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"

def to_timecode_frames_in(milliseconds, frame_rate):
    # 使用 Decimal 进行高精度运算
    total_seconds = Decimal(milliseconds) / Decimal(1000.0)

    # 针对不同帧率的计算
    if abs(frame_rate - 23.976) < 0.001:
        total_frames = total_seconds * Decimal(24000) / Decimal(1001)
        frames = math.floor(total_frames % 24)  # 取整帧
        total_seconds = math.floor(total_frames / 24)  # 计算总秒数
    elif abs(frame_rate - 24) < 0.001 or abs(frame_rate - 25) < 0.001 or abs(frame_rate - 50) < 0.001:
        total_frames = total_seconds * Decimal(frame_rate)
        frames = math.floor(total_frames % frame_rate)  # 取整帧
        total_seconds = int(total_seconds)  # 取整秒
    elif abs(frame_rate - 29.97) < 0.001:
        total_frames = total_seconds * Decimal(30000) / Decimal(1001)
        frames = math.floor(total_frames % 30)  # 取整帧
        total_seconds = math.floor(total_frames / 30)  # 计算总秒数
    elif abs(frame_rate - 59.94) < 0.001:
        total_frames = total_seconds * Decimal(60000) / Decimal(1001)
        frames = math.floor(total_frames % 60)  # 取整帧
        total_seconds = math.floor(total_frames / 60)  # 计算总秒数
    else:
        raise ValueError(f"Unsupported frame rate: {frame_rate}")

    # 计算小时、分钟和秒
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

def to_timecode_frames_out(milliseconds, frame_rate):
    # 使用 Decimal 进行高精度运算
    milliseconds = Decimal(milliseconds)
    
    # 针对不同帧率的计算
    if abs(frame_rate - 23.976) < 0.001:
        total_frames = math.floor(milliseconds * Decimal(24000) / Decimal(1001) / Decimal(1000))
        frames = total_frames % 24
        total_seconds = total_frames // 24
    elif abs(frame_rate - 24) < 0.001 or abs(frame_rate - 25) < 0.001 or abs(frame_rate - 50) < 0.001:
        total_frames = math.floor(milliseconds * Decimal(frame_rate) / Decimal(1000))
        frames = total_frames % round(frame_rate)
        total_seconds = total_frames // round(frame_rate)
    elif abs(frame_rate - 29.97) < 0.001:
        total_frames = math.floor(milliseconds * Decimal(30000) / Decimal(1001) / Decimal(1000))
        frames = total_frames % 30
        total_seconds = total_frames // 30
    elif abs(frame_rate - 59.94) < 0.001:
        total_frames = math.floor(milliseconds * Decimal(60000) / Decimal(1001) / Decimal(1000))
        frames = total_frames % 60
        total_seconds = total_frames // 60
    else:
        raise ValueError(f"Unsupported frame rate: {frame_rate}")

    # 计算小时、分钟和秒
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

def to_timecode_frames(milliseconds, frame_rate, is_out=False):
    if is_out:
        return to_timecode_frames_out(milliseconds, frame_rate)
    else:
        return to_timecode_frames_in(milliseconds, frame_rate)

def generate_srt(events, srt_output_file):
    with open(srt_output_file, 'w', encoding='utf-8') as srt_file:
        for i, event in enumerate(events):
            srt_file.write(f"{i+1}\n")
            srt_file.write(f"{event['start_srt']} --> {event['end_srt']}\n")
            srt_file.write(f"空\n\n")

def main(input_file, output_dir, srt_output_file=None):
    print(f"Starting Decoding Presentation Graphic Stream (PGS) File: {input_file}")
    pgs = PGSReader(input_file)

    images = []
    events = []
    timestamps_tc = []

    frame_rates = pgs.iter_segments()
    if not frame_rates:
        frame_rate = 24  # Default to 24 FPS if unknown
    else:
        frame_rate = next(iter(frame_rates))

    # Ensure frame_rate is a float
    frame_rate = float(frame_rate)

    # Format frame rates string
    frame_rates_str = ', '.join(f"{rate:.3f}" if rate % 1 != 0 else f"{int(rate)}" for rate in frame_rates)

    sorted_displaysets = sorted(pgs.displaysets, key=lambda ds: ds.pcs[0].presentation_timestamp if ds.pcs else 0)

    num_displaysets = len(sorted_displaysets)
    if num_displaysets == 0:
        print("Error: The .SUP File Is Incorrect Please Check The File.")
        return

    timestamps = []
    for i, ds in enumerate(sorted_displaysets):
        for j, pcs in enumerate(ds.pcs):
            timestamp = int(pcs.presentation_timestamp)
            timestamp_ms = format_timestamp(pcs.presentation_timestamp)
            timestamp_tc_ms = to_timecode_milliseconds(timestamp_ms)
            timestamp_tc_frames = to_timecode_frames(timestamp_ms, frame_rate)
            timestamps.append(pcs.presentation_timestamp)
            timestamps_tc.append(timestamp_ms)

    image_counter = 1
    for i, ds in enumerate(sorted_displaysets):
        if ds.has_image:
            pcs = ds.pcs[0]
            pds = ds.pds[0]
            ods = ds.ods[0]
            wds = ds.wds[0] if ds.wds else None

            pixels = read_rle_bytes(ods.img_data)
            img = create_image(ods.width, ods.height, pds.palette, pixels)
            images.append(img)

            x = wds.x_offset if wds else 0
            y = wds.y_offset if wds else 0
            width = ods.width
            height = ods.height

            start_timestamp_tc_frames = to_timecode_frames(timestamps_tc[i], frame_rate, is_out=False) if i < len(timestamps_tc) else "00:00:00:00"
            end_timestamp_tc_frames = to_timecode_frames(timestamps_tc[i+1], frame_rate, is_out=True) if (i+1) < len(timestamps_tc) else start_timestamp_tc_frames

            start_timestamp_tc_srt = to_timecode_milliseconds(format_timestamp(timestamps_tc[i])) if i < len(timestamps_tc) else "00:00:00,000"
            end_timestamp_tc_srt = to_timecode_milliseconds(format_timestamp(timestamps_tc[i+1])) if (i+1) < len(timestamps_tc) else start_timestamp_tc_srt

            final_path = os.path.join(output_dir, f'{image_counter:04d}.png')
            img.save(final_path)

            event = {
                'start': start_timestamp_tc_frames,
                'end': end_timestamp_tc_frames,
                'start_srt': start_timestamp_tc_srt,
                'end_srt': end_timestamp_tc_srt,
                'width': width,
                'height': height,
                'x': x,
                'y': y,
                'image': f'{image_counter:04d}.png',
                'frame_rates_str': frame_rates_str
            }
            events.append(event)

            image_counter += 1

    print(f"Saved {image_counter - 1} Image Files!")

    xml_output_file = os.path.join(output_dir, 'BDN_Script.xml')
    create_xml_script(events, xml_output_file, frame_rate)

    if srt_output_file:
        generate_srt(events, srt_output_file)
        print(f"Saved SubRip File: {srt_output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: PGS_Decoded_Tools_Plus 1.0.9 <Input_SUP_File> <Output_Images_Directory> [-srt]\n[Notice: Automatically Generate An BDN_Script.xml File In The Image Path]\n-srt: Specify The Path To Output SubRip File")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    srt_output_file = None

    if len(sys.argv) > 3 and sys.argv[3] == '-srt':
        if len(sys.argv) > 4:
            srt_output_file = sys.argv[4]
        else:
            print("Error: Please specify the output path for the SRT file when using the “-srt” option.")
            sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(input_file, output_dir, srt_output_file)
