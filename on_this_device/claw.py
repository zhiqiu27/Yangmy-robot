import serial
import time


def send_hex_to_com6(hex_data):
    try:
        # 配置串口
        ser = serial.Serial(
            port='/dev/wheeltec_mic',
            baudrate=115200,  # 波特率，可根据需要修改
            timeout=1
        )

        # 确保串口已打开
        if ser.is_open:
            print(f"已连接到 {ser.name}")

            # 发送十六进制数据
            ser.write(hex_data)
            print(f"已发送十六进制数据: {hex_data.hex(' ')}")

            # 等待片刻以确保发送完成
            time.sleep(0.1)

        # 关闭串口
        ser.close()

    except serial.SerialException as e:
        print(f"串口错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")


def get_user_input():
    # 预定义的两组数据
    data1 = bytes([0x7b, 0x01, 0x02, 0x01, 0x20, 0x49, 0x20, 0x00, 0xc8, 0xf8, 0x7d])
    data2 = bytes([0x7b, 0x01, 0x02, 0x00, 0x20, 0x49, 0x20, 0x00, 0xc8, 0xf9, 0x7d])

    print("请选择要发送的十六进制数据：")
    print("1: 7b 01 02 01 20 49 20 00 c8 f8 7d")
    print("2: 7b 01 02 00 20 46 50 01 2c 63 7d")
    print("3: 自定义输入")

    choice = input("请输入选项 (1, 2, 3): ").strip()

    if choice == '1':
        return data1
    elif choice == '2':
        return data2
    elif choice == '3':
        custom_input = input("请输入十六进制数据（例如：7b 01 02 ...）：").strip()
        try:
            # 将输入的十六进制字符串转换为字节
            hex_bytes = bytes.fromhex(custom_input)
            return hex_bytes
        except ValueError:
            print("错误：无效的十六进制格式！")
            return None
    else:
        print("错误：无效选项！")
        return None


def main():
    # 获取用户选择的十六进制数据
    hex_data = get_user_input()

    # 如果数据有效，则发送
    if hex_data:
        send_hex_to_com6(hex_data)
    else:
        print("未发送数据，请重试。")


if __name__ == "__main__":
    main()